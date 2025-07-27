# pipeline/dispatcher.py
"""
Pipeline消息分发器 - 支持NCCL直传优化
"""

import torch
import torch.distributed as dist
import logging
import threading
import queue
import time
from typing import List, Optional
from dataclasses import dataclass

from .message import Message
from .comm import Channel

logger = logging.getLogger(__name__)


def _init_dist(local_rank):
	"""保证 NCCL ready"""
	if not dist.is_initialized():
		dist.init_process_group(
			backend='nccl' if torch.cuda.is_available() else 'gloo')
	if torch.cuda.is_available():
		torch.cuda.set_device(local_rank)


@dataclass
class ChannelConfig:
	"""通道配置"""
	src_rank: int
	dst_rank: int
	src_device: torch.device
	dst_device: torch.device
	buffer_size: int = 10


class Dispatcher:
	"""
	Pipeline消息分发器

	职责:
	1. 管理Stage间的消息路由
	2. 处理跨GPU通信 (NCCL/Gloo)
	3. 缓冲区管理和流控
	4. 设备对齐和tensor传输优化
	"""
	
	def __init__(self, stage, rank: int, in_channels: List[Channel] = None,
	             out_channels: List[Channel] = None):
		"""
		Args:
			stage: 当前Stage实例
			rank: 分布式rank
			in_channels: 输入通道列表
			out_channels: 输出通道列表
		"""
		self.stage = stage
		self.rank = rank
		self.in_channels = in_channels or []
		self.out_channels = out_channels or []
		
		# 初始化分布式
		_init_dist(self.rank)
		
		# 消息缓冲区
		self.input_queue = queue.Queue(maxsize=100)
		self.output_queue = queue.Queue(maxsize=100)
		
		# 控制标志
		self.running = False
		self.shutdown_event = threading.Event()
		
		# 统计信息
		self.processed_count = 0
		self.error_count = 0
		
		logger.info(f"Dispatcher初始化完成: rank={rank}, "
		            f"in_channels={len(self.in_channels)}, "
		            f"out_channels={len(self.out_channels)}")
	
	def start(self):
		"""启动dispatcher"""
		if self.running:
			logger.warning("Dispatcher已在运行")
			return
		
		self.running = True
		self.shutdown_event.clear()
		
		# 启动处理线程
		self.processing_thread = threading.Thread(target=self._run_processing_loop)
		self.processing_thread.daemon = True
		self.processing_thread.start()
		
		# 启动接收线程
		if self.in_channels:
			self.receiving_thread = threading.Thread(target=self._run_receiving_loop)
			self.receiving_thread.daemon = True
			self.receiving_thread.start()
		
		logger.info(f"Dispatcher启动: rank={self.rank}")
	
	def stop(self):
		"""停止dispatcher"""
		if not self.running:
			return
		
		self.running = False
		self.shutdown_event.set()
		
		# 等待线程结束
		if hasattr(self, 'processing_thread'):
			self.processing_thread.join(timeout=5.0)
		if hasattr(self, 'receiving_thread'):
			self.receiving_thread.join(timeout=5.0)
		
		logger.info(f"Dispatcher停止: rank={self.rank}")
	
	def send_message(self, msg: Message):
		"""发送消息到输出队列"""
		try:
			self.output_queue.put(msg, timeout=1.0)
		except queue.Full:
			logger.warning(f"输出队列满，丢弃消息: {msg.kind}")
			self.error_count += 1
	
	def _run_receiving_loop(self):
		"""接收消息循环"""
		while self.running and not self.shutdown_event.is_set():
			try:
				for channel in self.in_channels:
					# 非阻塞接收
					try:
						# 简化版接收，实际需要根据通信协议实现
						msg_data = dist.recv_object(src=channel.src_rank, timeout=0.1)
						msg = Message.from_dict(msg_data)
						
						# 使进入 Stage 前张量落到本 GPU
						if self.stage.device.type == 'cuda':
							msg.cuda_()
						
						self.input_queue.put(msg, timeout=0.1)
					except Exception:
						# 非阻塞，没有消息时继续
						continue
				
				time.sleep(0.001)  # 短暂休眠避免CPU占用过高
			
			except Exception as e:
				logger.error(f"接收消息失败: {e}")
				self.error_count += 1
	
	def _run_processing_loop(self):
		"""处理消息循环"""
		while self.running and not self.shutdown_event.is_set():
			try:
				# 获取输入消息
				try:
					input_msg = self.input_queue.get(timeout=0.1)
				except queue.Empty:
					continue
				
				# 使进入 Stage 前张量落到本 GPU（备用检查）
				if self.stage.device.type == 'cuda':
					input_msg.cuda_()
				
				# Stage处理
				try:
					output_messages = list(self.stage.process(input_msg))
					self.processed_count += 1
				except Exception as e:
					logger.error(f"Stage处理失败: {e}")
					self.error_count += 1
					continue
				
				# 发送输出消息
				for msg in output_messages:
					self._send_message(msg)
			
			except Exception as e:
				logger.error(f"处理循环错误: {e}")
				self.error_count += 1
	
	def _send_message(self, msg: Message):
		"""发送消息到下游Stage"""
		if not self.out_channels:
			return
		
		success_count = 0
		
		for channel in self.out_channels:
			try:
				# --- 设备对齐 ---
				device_msg = msg.cuda_() if channel.dst_device.type == 'cuda' else msg.cpu_()
				
				# --- 发送协议 ---
				if device_msg.device.type == 'cuda':
					# 直发张量：dst_device.rank 与 channel.dst_rank 对齐
					# 简化版：只发送主要tensor，实际需要完整协议
					if 'image' in device_msg.payload:
						dist.isend(device_msg.payload['image'], dst=channel.dst_rank)
					elif 'features' in device_msg.payload:
						dist.isend(device_msg.payload['features'], dst=channel.dst_rank)
					elif 'logits' in device_msg.payload:
						dist.isend(device_msg.payload['logits'], dst=channel.dst_rank)
				else:
					# fallback: 对象发送 (gloo)
					dist.send_object(device_msg.to_dict(), dst=channel.dst_rank)
				
				success_count += 1
			
			except Exception as e:
				logger.error(f"Send error via {channel}: {e}")
				self.error_count += 1
		
		if success_count == 0:
			logger.warning(f"消息发送完全失败: {msg.kind}")
	
	def get_stats(self):
		"""获取统计信息"""
		return {
			'rank': self.rank,
			'processed_count': self.processed_count,
			'error_count': self.error_count,
			'input_queue_size': self.input_queue.qsize(),
			'output_queue_size': self.output_queue.qsize(),
			'running': self.running
		}
	
	def __enter__(self):
		self.start()
		return self
	
	def __exit__(self, exc_type, exc_val, exc_tb):
		self.stop()


