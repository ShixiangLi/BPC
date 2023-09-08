import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function, Variable

__all__ = ['BinarizeConv1d', 'BinarizeLinear']


class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


class BinaryQuantize_a(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.where(input >= 0.0, torch.tensor(1.0), torch.tensor(0.0))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (2 - torch.abs(2 * input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input


class BinarizeConv1d(nn.Conv1d):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv1d, self).__init__(*kargs, **kwargs)
        self.alpha = nn.Parameter(torch.rand(self.weight.size(0), 1), requires_grad=True)

    def forward(self, input):
        a = input
        w = self.weight

        w0 = w - w.mean([1], keepdim=True)

        # * binarize
        bw = BinaryQuantize().apply(w0)
        ba = BinaryQuantize_a().apply(a)

        # * scale
        ow = bw * (torch.sum(torch.abs(w0), dim=[1], keepdim=True) / torch.sum(torch.abs(bw), dim=[1], keepdim=True))
        oa = ba * (torch.sum(torch.abs(a), dim=[1], keepdim=True) / torch.sum(torch.abs(a), dim=[1], keepdim=True))

        # * 1bit conv
        output = F.conv1d(a, ow, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # * scaling factor
        output = output
        return output


class BinarizeLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)
        self.alpha = nn.Parameter(torch.rand(self.weight.size(0)), requires_grad=True)

    def forward(self, input):
        a = input
        w = self.weight
        w0 = w - w.mean([1], keepdim=True)

        # * binarize
        bw = BinaryQuantize().apply(w0)
        ba = BinaryQuantize_a().apply(a)

        # * scale
        ow = bw * (torch.sum(torch.abs(w0), dim=[1], keepdim=True) / torch.sum(torch.abs(bw), dim=[1], keepdim=True))
        oa = ba * (torch.sum(torch.abs(a), dim=[1], keepdim=True) / torch.sum(torch.abs(a), dim=[1], keepdim=True))
        # * 1bit conv
        output = F.linear(a, ow, self.bias)
        # * scaling factor
        output = output
        return output
