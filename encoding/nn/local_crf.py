##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Xiaofeng Ding
## Email: dxfeng@shu.edu.cn
## Copyright (c) 2020
##
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Local CRF"""
import warnings

import torch
import torch.nn as nn
from torch.nn.modules import Module
from ..functions import *
from torch.nn import init
from torch.nn import functional as F


__all__ = ['LocalCRF', 'GACRF']

class LocalCRF(Module):
    def __init__(self, num_category, learning_compatibility=True):
        super(LocalCRF, self).__init__()
        self.num_category = num_category
        self.learning_compatibility = learning_compatibility
        compatibility_shape = [num_category, num_category, 1, 1]
        if learning_compatibility:
            #compatibility = torch.ones(compatibility_shape) * 0.02 - torch.eye(num_category, num_category).reshape(compatibility_shape) * 0.04
            compatibility = torch.zeros(compatibility_shape)
            self.compatibility = nn.Parameter(compatibility)
            self.reset_parameters()
        else:
            compatibility = torch.ones(compatibility_shape) - torch.eye(num_category, num_category).reshape(compatibility_shape)
            self.compatibility = nn.Parameter(compatibility, requires_grad=False)


    def reset_parameters(self):
        init.uniform(self.compatibility, -5e-2, 5e-2)
        #init.zeros_(self.compatibility)

    def forward(self, input_F, input_logit):
        # input_F: b, k, h, w
        # input_U: b, nclass, h, w
        # input_Q = softmax(input_logit)
        assert self.num_category == input_logit.shape[1]
        input_Q = torch.softmax(input_logit, dim=1)
        kernel_size = input_F.shape[1]
        kernel_len = int(math.sqrt(kernel_size))
        assert kernel_size == kernel_len ** 2

        # message passing
        output = local_crf(input_F, input_Q)

        # label compatibility
        if self.learning_compatibility:
            compatibility = F.sigmoid(100 * self.compatibility)
        else:
            compatibility = self.compatibility
        output = F.conv2d(output, compatibility)
        #print(compatibility[:,:,0,0])

        # local update
        output_U = input_logit - output
        return output_U

    def extra_repr(self):
        return 'num_category={}, learning_compatibility={}'.format(self.num_category, self.learning_compatibility)

class GACRF(Module):
    def __init__(self, num_category, num_group):
        super(GACRF, self).__init__()
        self.num_category = num_category
        self.num_group = num_group
        self.matrix = nn.Parameter(torch.zeros([num_group, num_category, 1, 1]))
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform(self.matrix, -5e-2, 5e-2)
        #init.uniform(self.matrix, -2e-1, 2e-1)



    def forward(self, input_F, input_logit):
        # input_F: b, k, h, w
        # input_U: b, nclass, h, w
        # input_Q = softmax(input_logit)
        b, num_category, h, w = input_logit.shape
        assert self.num_category == num_category
        input_Q = torch.softmax(input_logit, dim=1)
        kernel_size = input_F.shape[1]
        kernel_len = int(math.sqrt(kernel_size))
        assert kernel_size == kernel_len ** 2

        encoding_matrix = torch.softmax(100 * self.matrix, dim=0)
        input_Q = F.conv2d(input_Q, encoding_matrix)

        # message passing
        output = local_crf(input_F, input_Q)

        # label compatibility
        decoding_matrix = encoding_matrix.permute(1, 0, 2, 3)
        output = F.conv2d(output, decoding_matrix)

        # local update
        output_U = input_logit - output
        return output_U

    def extra_repr(self):
        return 'num_category={}, num_group={}'.format(self.num_category, self.num_group)