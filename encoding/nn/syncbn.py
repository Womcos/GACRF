##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Synchronized Cross-GPU Batch Normalization Module"""
import warnings
try:
    from queue import Queue
except ImportError:
    from Queue import Queue

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from ..utils.misc import EncodingDeprecationWarning
from ..functions import *


__all__ = ['SyncBatchNorm']


class SyncBatchNorm(_BatchNorm):
    r"""Cross-GPU Synchronized Batch normalization (SyncBN)

    Standard BN [1]_ implementation only normalize the data within each device (GPU).
    SyncBN normalizes the input within the whole mini-batch.
    We follow the sync-onece implmentation described in the paper [2]_ .
    Please see the design idea in the `notes <./notes/syncbn.html>`_.

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    The mean and standard-deviation are calculated per-channel over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        sync: a boolean value that when set to ``True``, synchronize across
            different gpus. Default: ``True``
        activation : str
            Name of the activation functions, one of: `leaky_relu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> m = SyncBatchNorm(100)
        >>> net = torch.nn.DataParallel(m)
        >>> output = net(input)
        >>> # for Inpace ABN
        >>> ABN = partial(SyncBatchNorm, activation='leaky_relu', slope=0.01, sync=True, inplace=True)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, sync=True, activation="none", slope=0.01,
                 inplace=True):
        super(SyncBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=True)
        self.activation = activation
        self.inplace = False if activation == 'none' else inplace
        #self.inplace = inplace
        self.slope = slope
        self.devices = list(range(torch.cuda.device_count()))
        self.sync = sync if len(self.devices) > 1 else False
        # Initialize queues
        self.worker_ids = self.devices[1:]
        self.master_queue = Queue(len(self.worker_ids))
        self.worker_queues = [Queue(1) for _ in self.worker_ids]
        # running_exs
        #self.register_buffer('running_exs', torch.ones(num_features))

    def _check_input_dim(self, x):
        pass

    def forward(self, x):
        if not self.training:
            return super().forward(x)
        # Resize the input to (B, C, -1).
        input_shape = x.size()
        x = x.view(input_shape[0], self.num_features, -1)
        if x.get_device() == self.devices[0]:
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
                "worker_queue": self.worker_queues[self.worker_ids.index(x.get_device())]
            }
        return syncbatchnorm(x, self.weight, self.bias, self.running_mean, self.running_var,
                                 extra, self.sync, self.training, self.momentum, self.eps,
                                 self.activation, self.slope).view(input_shape)

    def extra_repr(self):
        if self.activation == 'none':
            return 'sync={}'.format(self.sync)
        else:
            return 'sync={}, act={}, slope={}, inplace={}'.format(
                self.sync, self.activation, self.slope, self.inplace
            )