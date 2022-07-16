##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Xiaofeng Ding
## Email: dxfeng@shu.edu.cn
## Copyright (c) 2021
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import division

import warnings
try:
    from queue import Queue
except ImportError:
    from Queue import Queue

import torch
import torch.nn as nn
from torch.nn.modules import Module
from ..functions import *
from torch.nn import init


__all__ = ['CategoryCenter', 'SyncCategoryCenter', 'SyncCategoryCenter2', 'Memory']

class CategoryCenter(Module):
    def __init__(self, num_features, num_category, eps=1e-5, momentum=0.1, affine=True):
        super(CategoryCenter, self).__init__()
        self.num_features = num_features
        self.num_category = num_category
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.register_buffer('running_center', torch.zeros([num_category, num_features]))
        if self.affine:
            self.bias = nn.Parameter(torch.zeros([num_category, num_features]))
            self.weight = nn.Parameter(torch.ones([num_category, num_features]))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_center.zero_()
        if self.affine:
            init.zeros_(self.bias)
            init.ones_(self.weight)

    def forward(self, input_feat, input_mask):
        # input_feat: b, nfeat, h, w
        # input_mask (probability map): b, nclass, h, w
        assert self.num_category == input_mask.shape[1]
        assert self.num_features == input_feat.shape[1]
        b, _, h, w = input_mask.shape
        input_mask = input_mask.permute(1, 0, 2, 3).contiguous().view(self.num_category, -1)  # nclass, bhw
        if self.training:
            input_feat = input_feat.permute(1, 0, 2, 3).contiguous().view(self.num_features, -1)   # nfeat, bhw
            input_feat = input_feat.permute(1, 0)   # bhw, nfeat
            mask_sum = torch.sum(input_mask, dim=-1, keepdim=True)  # nclass, 1
            mask_max = input_mask.max(-1)[0].view(self.num_category, 1)    # nclass, 1
            feat_center = torch.matmul(input_mask, input_feat) / (mask_sum + self.eps)   # nclass, nfeat
            momentum = self.momentum * mask_max   # nclass, 1
            running_center = self.running_center * (1. - momentum) + momentum * feat_center
            self.running_center.copy_(running_center)
        else:
            feat_center = self.running_center

        if self.affine:
            feat_center = feat_center * self.weight + self.bias
        out_feat = torch.matmul(feat_center.permute(1, 0), input_mask).view(self.num_features, b, h, w)
        out_feat = out_feat.permute(1, 0, 2, 3)  # b, nfeat, h, w
        return out_feat

class SyncCategoryCenter(CategoryCenter):
    def __init__(self, num_features, num_category, eps=1e-5, momentum=0.1, affine=True, sync=True):
        super(SyncCategoryCenter, self).__init__(num_features, num_category, eps=eps, momentum=momentum, affine=affine)
        self.devices = list(range(torch.cuda.device_count()))
        self.sync = sync if len(self.devices) > 1 else False
        # Initialize queues
        self.worker_ids = self.devices[1:]
        self.master_queue = Queue(len(self.worker_ids))
        self.worker_queues = [Queue(1) for _ in self.worker_ids]

    def forward(self, input_feat, input_mask):
        if not self.training:
            return super().forward(input_feat, input_mask)
        b, _, h, w = input_mask.shape
        input_mask = input_mask.permute(1, 0, 2, 3).contiguous().view(self.num_category, -1)  # nclass, bhw
        input_feat = input_feat.permute(1, 0, 2, 3).contiguous().view(self.num_features, -1)  # nfeat, bhw
        input_feat = input_feat.permute(1, 0)  # bhw, nfeat
        feat_sum = torch.matmul(input_mask, input_feat)  # nclass, nfeat
        mask_sum = torch.sum(input_mask, dim=-1, keepdim=True)  # nclass, 1
        mask_max = input_mask.max(-1)[0].view(self.num_category, 1)  # nclass, 1

        if input_feat.get_device() == self.devices[0]:
            # Master mode
            extra = {
                "is_master": True,
                "master_queue": self.master_queue,
                "worker_queues": self.worker_queues,
                "worker_ids": self.worker_ids
            }
        else:
            # Worker mode
            extra = {
                "is_master": False,
                "master_queue": self.master_queue,
                "worker_queue": self.worker_queues[self.worker_ids.index(input_feat.get_device())]
            }

        feat_center = sync_category_center(feat_sum, mask_sum, mask_max, self.running_center,
                             extra, self.momentum, self.eps)

        if self.affine:
            feat_center = feat_center * self.weight + self.bias
        out_feat = torch.matmul(feat_center.permute(1, 0), input_mask).view(self.num_features, b, h, w)
        out_feat = out_feat.permute(1, 0, 2, 3)  # b, nfeat, h, w
        return out_feat

    def extra_repr(self):
        return 'sync={}'.format(self.sync)

class SyncCategoryCenter2(CategoryCenter):
    # add the local center
    def __init__(self, num_features, num_category, eps=1e-5, momentum=0.1, affine=True, sync=True):
        super(SyncCategoryCenter2, self).__init__(num_features, num_category, eps=eps, momentum=momentum, affine=affine)
        self.devices = list(range(torch.cuda.device_count()))
        self.sync = sync if len(self.devices) > 1 else False
        # Initialize queues
        self.worker_ids = self.devices[1:]
        self.master_queue = Queue(len(self.worker_ids))
        self.worker_queues = [Queue(1) for _ in self.worker_ids]

    def forward(self, input_feat, input_mask):
        b, _, h, w = input_mask.shape
        input_mask = input_mask.view(b, self.num_category, -1)  # b, nclass, hw
        input_feat = input_feat.view(b, self.num_features, -1)  # b, nfeat, hw
        input_feat = input_feat.permute(0, 2, 1)  # b, hw, nfeat

        #  image level center
        feat_sum_img = torch.matmul(input_mask, input_feat)  # b, nclass, nfeat
        mask_sum_img = torch.sum(input_mask, dim=-1, keepdim=True)  # b, nclass, 1
        feat_center_img = feat_sum_img / (mask_sum_img + self.eps)   # b, nclass, nfeat
        feat_center_img = feat_center_img.permute(0, 2, 1)  # b, nfeat, nclass
        out_feat_img = torch.matmul(feat_center_img, input_mask).view(b, self.num_features, h, w)

        #  dataset level center
        if self.training:
            feat_sum = torch.sum(feat_sum_img, dim=0)  # nclass, nfeat
            mask_sum = torch.sum(mask_sum_img, dim=0)  # nclass, 1
            mask_max = input_mask.max(0)[0].max(-1)[0].view(self.num_category, 1)  # nclass, 1
            if input_feat.get_device() == self.devices[0]:
                # Master mode
                extra = {
                    "is_master": True,
                    "master_queue": self.master_queue,
                    "worker_queues": self.worker_queues,
                    "worker_ids": self.worker_ids
                }
            else:
                # Worker mode
                extra = {
                    "is_master": False,
                    "master_queue": self.master_queue,
                    "worker_queue": self.worker_queues[self.worker_ids.index(input_feat.get_device())]
                }

            feat_center = sync_category_center(feat_sum, mask_sum, mask_max, self.running_center,
                                 extra, self.momentum, self.eps)
        else:
            feat_center = self.running_center

        if self.affine:
            feat_center = feat_center * self.weight + self.bias
        input_mask = input_mask.permute(1, 0, 2).contiguous().view(self.num_category, -1)
        out_feat = torch.matmul(feat_center.permute(1, 0), input_mask).view(self.num_features, b, h, w)
        out_feat = out_feat.permute(1, 0, 2, 3)  # b, nfeat, h, w
        return torch.cat([out_feat_img, out_feat], dim=1)

    def extra_repr(self):
        return 'sync={}'.format(self.sync)

class Memory(Module):
    def __init__(self, num_features, num_category, num_memory):
        super(Memory, self).__init__()
        self.num_features = num_features
        self.num_category = num_category
        self.num_memory = num_memory
        self.memory_feat = nn.Parameter(torch.zeros([num_category, num_memory, num_features]))
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform(self.memory_feat)

    def forward(self, input_feat, input_mask, efficient_inference=False):
        # input_feat: b, nfeat, h, w
        # input_mask (probability map): b, nclass, h, w
        assert self.num_category == input_mask.shape[1]
        assert self.num_features == input_feat.shape[1]
        b, _, h, w = input_mask.shape
        if self.training or not efficient_inference:
            memory_feat = self.memory_feat.view(-1, self.num_features)  # nclass*nmem, nfeat
            input_mask = input_mask.permute(1, 0, 2, 3).contiguous().view(self.num_category, 1, -1)  # nclass, 1 bhw
            input_feat = input_feat.permute(1, 0, 2, 3).contiguous().view(self.num_features, -1)  # nfeat, bhw
            #memory_feat_norm = torch.norm(memory_feat, dim=1, keepdim=True)
            #input_feat_norm = torch.norm(input_feat, dim=0, keepdim=True)
            similarity = torch.matmul(memory_feat, input_feat)  # nclass*nmem, bhw
            similarity = similarity.view(self.num_category, self.num_memory, -1)  # nclass, nmem, bhw
            weight = torch.softmax(similarity, dim=1)  # nclass, nmem, bhw
            weight = weight * input_mask
            weight = weight.view(self.num_category * self.num_memory, -1).permute(1, 0)  # bhw, nclass*nmem
            out_feat = torch.matmul(weight, memory_feat).view(b, h, w, self.num_features).permute(0, 3, 1, 2)
        else:
            input_mask = torch.argmax(input_mask, 1).view(-1)
            memory_feat = torch.index_select(self.memory_feat, 0, input_mask)  # bhw, nmen, nfeat
            input_feat = input_feat.permute(0, 2, 3, 1).contiguous().view(-1, self.num_features, 1)  # bhw, nfeat, 1
            similarity = torch.matmul(memory_feat, input_feat)  # bhw, nmen, 1
            weight = torch.softmax(similarity, dim=1)  # bhw, nmen, 1
            weight = weight.view(-1, 1, self.num_memory)  # bhw, 1, nmen
            out_feat = torch.matmul(weight, memory_feat).view(b, h, w, self.num_features).permute(0, 3, 1, 2)
        return out_feat

    def infer_weight(self, input_feat, input_mask):
        # input_feat: b, nfeat, h, w
        # input_mask (probability map): b, nclass, h, w
        assert self.num_category == input_mask.shape[1]
        assert self.num_features == input_feat.shape[1]
        b, _, h, w = input_mask.shape
        input_mask = torch.argmax(input_mask, 1).view(-1)
        memory_feat = torch.index_select(self.memory_feat, 0, input_mask)  # bhw, nmen, nfeat
        input_feat = input_feat.permute(0, 2, 3, 1).contiguous().view(-1, self.num_features, 1)  # bhw, nfeat, 1
        similarity = torch.matmul(memory_feat, input_feat)  # bhw, nmen, 1
        #weight = torch.softmax(similarity, dim=1)  # bhw, nmen, 1
        weight = similarity.view(-1, 1, self.num_memory).view(b, h, w, self.num_memory).permute(0, 3, 1, 2)
        return weight
