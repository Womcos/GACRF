##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Xiaofeng Ding
## Email: dxfeng@shu.edu.cn
## Copyright (c) 2021
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
import torch.cuda.comm as comm
from torch.autograd import Function
from torch.autograd.function import once_differentiable

__all__ = ['sync_category_center']


class sync_category_center_(Function):
    @classmethod
    def forward(cls, ctx, feat_sum_input, mask_sum_input, mask_max_input, running_center,
                extra, momentum=0.1, eps=1e-05):
        # save context
        cls._parse_extra(ctx, extra)
        ctx.momentum = momentum
        ctx.eps = eps

        # continous inputs
        feat_sum_input = feat_sum_input.contiguous()
        mask_sum_input = mask_sum_input.contiguous()
        mask_max_input = mask_max_input.contiguous()

        if ctx.is_master:
            feat_sum, mask_sum, mask_max = [feat_sum_input.unsqueeze(0)], [mask_sum_input.unsqueeze(0)], [mask_max_input.unsqueeze(0)]
            for _ in range(ctx.master_queue.maxsize):
                feat_sum_w, mask_sum_w, mask_max_w = ctx.master_queue.get()
                ctx.master_queue.task_done()
                feat_sum.append(feat_sum_w.unsqueeze(0))
                mask_sum.append(mask_sum_w.unsqueeze(0))
                mask_max.append(mask_max_w.unsqueeze(0))

            feat_sum = torch.cat(feat_sum, dim=0).sum(0)    # nclass, nfeat
            mask_sum = torch.cat(mask_sum, dim=0).sum(0)    # nclass, 1
            mask_max = torch.cat(mask_max, dim=0).max(0)[0]  # nclass, 1
            # calculate the feature center
            feat_center = feat_sum / (mask_sum + eps)  # nclass, nfeat

            # update the running mean
            momentum = momentum * mask_max  # nclass, 1
            running_center.copy_(running_center * (1. - momentum) + momentum * feat_center)

            # broadcast the feature center to all devices
            tensors = comm.broadcast(feat_center, [feat_sum.get_device()] + ctx.worker_ids)
            for ts, queue in zip(tensors[1:], ctx.worker_queues):
                queue.put(ts)

            # save parameters for backward
            ctx.save_for_backward(feat_sum, mask_sum)
        else:
            ctx.master_queue.put((feat_sum_input, mask_sum_input, mask_max_input))
            feat_center = ctx.worker_queue.get()
            ctx.worker_queue.task_done()
        return feat_center

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        dz = dz.contiguous()
        if ctx.is_master:
            feat_sum, mask_sum = ctx.saved_tensors
            d_feat_center = [dz.unsqueeze(0)]
            for _ in range(ctx.master_queue.maxsize):
                d_feat_center_w = ctx.master_queue.get()
                ctx.master_queue.task_done()
                d_feat_center.append(d_feat_center_w.unsqueeze(0))

            # calculate the gradient
            d_feat_center = torch.cat(d_feat_center, dim=0).sum(0)  # nclass, nfeat
            d_feat_sum = d_feat_center / (mask_sum + ctx.eps)   # nclass, nfeat
            d_mask_sum = - d_feat_center * feat_sum / torch.pow(mask_sum + ctx.eps, 2)  # nclass, nfeat
            d_mask_sum = d_mask_sum.sum(-1).view(-1, 1)  # nclass, 1

            # broadcast
            tensors = comm.broadcast_coalesced((d_feat_sum, d_mask_sum), [d_feat_sum.get_device()] + ctx.worker_ids)
            for ts, queue in zip(tensors[1:], ctx.worker_queues):
                queue.put(ts)
        else:
            ctx.master_queue.put(dz)
            d_feat_sum, d_mask_sum = ctx.worker_queue.get()
            ctx.worker_queue.task_done()
        return d_feat_sum, d_mask_sum, None, None, None, None, None

    @staticmethod
    def _parse_extra(ctx, extra):
        ctx.is_master = extra["is_master"]
        if ctx.is_master:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queues = extra["worker_queues"]
            ctx.worker_ids = extra["worker_ids"]
        else:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queue = extra["worker_queue"]

sync_category_center = sync_category_center_.apply