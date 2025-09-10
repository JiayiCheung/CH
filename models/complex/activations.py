
import torch
import torch.nn.functional as F


def relu(input, inplace=False):
    """复数兼容的ReLU"""
    if torch.is_complex(input):
        if inplace and input.is_leaf:
            raise RuntimeError("修改具有梯度的复数张量的.real或.imag属性不支持原地操作")
        return torch.complex(F.relu(input.real, inplace=False),
                             F.relu(input.imag, inplace=False))
    else:
        return F.relu(input, inplace=inplace)


def sigmoid(input):
    """复数兼容的Sigmoid"""
    if torch.is_complex(input):
        return torch.complex(F.sigmoid(input.real), F.sigmoid(input.imag))
    else:
        return F.sigmoid(input)


def tanh(input):
    """复数兼容的Tanh"""
    if torch.is_complex(input):
        return torch.complex(F.tanh(input.real), F.tanh(input.imag))
    else:
        return F.tanh(input)