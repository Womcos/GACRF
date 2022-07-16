##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Xiaofeng Ding
## Email: dxfeng@shu.edu.cn
## Copyright (c) 2020
##
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Local diff"""
import torch
import torch.cuda.comm as comm
from torch.autograd import Function
from torch.autograd.function import once_differentiable

if torch.cuda.device_count() > 0:
    from encoding import gpu

import torch.nn.functional as F
import math

class local_diff_(Function):
    @classmethod
    def forward(cls, ctx, input_F, kernel_len):
        ctx.kernel_len = kernel_len
        # continous inputs
        input_F = input_F.contiguous()

        kernel_size = kernel_len ** 2
        pad_len = int((kernel_len - 1) / 2)
        assert kernel_size == kernel_len ** 2

        input_F = F.pad(input_F, (pad_len, pad_len, pad_len, pad_len), "constant", 0)

        output = gpu.local_diff_forward(input_F, kernel_len)

        # Output
        ctx.save_for_backward(input_F)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        input_F, = ctx.saved_tensors
        dz = dz.contiguous()

        kernel_len = ctx.kernel_len
        pad_len = int((kernel_len - 1) / 2)
        dz = F.pad(dz, (pad_len, pad_len, pad_len, pad_len), "constant", 0)

        # backward
        d_input_F = gpu.local_diff_backward(dz, input_F, kernel_len)
        d_input_F = d_input_F[:, :, pad_len:-pad_len, pad_len:-pad_len]
        return d_input_F, None



local_diff = local_diff_.apply