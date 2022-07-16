##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Xiaofeng Ding
## Email: dxfeng@shu.edu.cn
## Copyright (c) 2020
##
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Local CRF"""
import torch
import torch.cuda.comm as comm
from torch.autograd import Function
from torch.autograd.function import once_differentiable

if torch.cuda.device_count() > 0:
    from encoding import gpu

import torch.nn.functional as F
import math

class local_crf_(Function):
    @classmethod
    def forward(cls, ctx, input_F, input_Q):
        # continous inputs
        input_F = input_F.contiguous()
        input_Q = input_Q.contiguous()

        kernel_size = input_F.shape[1]
        kernel_len = int(math.sqrt(kernel_size))
        pad_len = int((kernel_len - 1) / 2)
        assert kernel_size == kernel_len ** 2
        ctx.kernel_size = kernel_size
        ctx.kernel_len = kernel_len

        input_F = F.pad(input_F, (pad_len, pad_len, pad_len, pad_len), "constant", 0)
        input_Q = F.pad(input_Q, (pad_len, pad_len, pad_len, pad_len), "constant", 0)

        output = gpu.local_crf_forward(input_F, input_Q)

        # Output
        ctx.save_for_backward(input_F, input_Q)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        input_F, input_Q = ctx.saved_tensors
        dz = dz.contiguous()

        kernel_len = ctx.kernel_len
        pad_len = int((kernel_len - 1) / 2)
        dz = F.pad(dz, (pad_len, pad_len, pad_len, pad_len), "constant", 0)

        # backward
        d_input_F, d_input_Q = gpu.local_crf_backward(dz, input_F, input_Q)

        d_input_F = d_input_F[:, :, pad_len:-pad_len, pad_len:-pad_len]
        d_input_Q = d_input_Q[:, :, pad_len:-pad_len, pad_len:-pad_len]
        return d_input_F, d_input_Q



local_crf = local_crf_.apply