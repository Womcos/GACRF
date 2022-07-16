##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Xiaofeng Ding
## Email: dxfeng@shu.edu.cn
## Copyright (c) 2020
##
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Local CRF2"""
import warnings

import torch
import torch.nn as nn
from torch.nn.modules import Module
from ..functions import *
from torch.nn import init
from torch.nn import functional as F


__all__ = ['LocalCRF2']

class LocalCRF2(Module):
    def __init__(self, num_category):
        super(LocalCRF2, self).__init__()
        self.num_category = num_category
        self.weight = nn.Parameter(torch.tensor(1., dtype=torch.float32))
        self.compatibility = nn.Parameter(torch.zeros([num_category, num_category, 1, 1]))
        self.bias = nn.Parameter(torch.zeros(num_category))
        self.reset_parameters()

    def reset_parameters(self):
        init.zeros_(self.compatibility)
        init.zeros_(self.bias)


    def forward(self, input_F, input_logit, kernel_len):
        # input_F: b, c, h, w
        # input_U: b, nclass, h, w
        # input_Q = softmax(input_logit)
        assert self.num_category == input_logit.shape[1]
        channel = input_F.shape[1]
        input_Q = torch.softmax(input_logit, dim=1)
        kernel_size = kernel_len ** 2
        assert kernel_size == kernel_len ** 2

        # input_F
        Diff = local_diff(input_F, kernel_len) / channel  # [b, kernel_size, h, w]
        input_F = self.weight * torch.exp(- Diff)   # [b, kernel_size, h, w]
        # message passing
        output = local_crf(input_F, input_Q) / kernel_len    # [b, nclass, h, w]

        # label compatibility
        compatibility = F.sigmoid(self.compatibility)
        output = F.conv2d(output, compatibility, self.bias)

        # local update
        output_U = input_logit - output
        return output_U

    def extra_repr(self):
        return 'num_category={}'.format(self.num_category)

