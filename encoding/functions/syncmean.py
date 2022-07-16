##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Xiaofeng Ding
## Email: dxfeng@shu.edu.cn
## Copyright (c) 2021
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Synchronized Cross-GPU functions"""
import torch
import torch.cuda.comm as comm
from torch.autograd import Function
from torch.autograd.function import once_differentiable


__all__ = ['syncmean']

class syncmean_(Function):
    @classmethod
    def forward(cls, ctx, x, extra):
        cls._parse_extra(ctx, extra)
        x = x.contiguous()
        if ctx.is_master:
            _x = [x.unsqueeze(0)]
            for _ in range(ctx.master_queue.maxsize):
                _x_w = ctx.master_queue.get()
                ctx.master_queue.task_done()
                _x.append(_x_w.unsqueeze(0))
            _x = comm.gather(_x).mean(0)

            # broadcast the feature center to all devices
            tensors = comm.broadcast(_x, [_x.get_device()] + ctx.worker_ids)
            for ts, queue in zip(tensors[1:], ctx.worker_queues):
                queue.put(ts)
        else:
            ctx.master_queue.put(x)
            _x = ctx.worker_queue.get()
            ctx.worker_queue.task_done()
        return _x

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        dz = dz.contiguous()

        if ctx.is_master:
            _dx = [dz.unsqueeze(0)]
            for _ in range(ctx.master_queue.maxsize):
                _dx_w = ctx.master_queue.get()
                ctx.master_queue.task_done()
                _dx.append(_dx_w.unsqueeze(0))
            _dx = comm.gather(_dx).mean(0)
            tensors = comm.broadcast(_dx, [_dx.get_device()] + ctx.worker_ids)
            for ts, queue in zip(tensors[1:], ctx.worker_queues):
                queue.put(ts)
        else:
            ctx.master_queue.put(dz)
            _dx = ctx.worker_queue.get()
            ctx.worker_queue.task_done()
        return _dx, None


    def _parse_extra(ctx, extra):
        ctx.is_master = extra["is_master"]
        if ctx.is_master:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queues = extra["worker_queues"]
            ctx.worker_ids = extra["worker_ids"]
        else:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queue = extra["worker_queue"]

syncmean = syncmean_.apply