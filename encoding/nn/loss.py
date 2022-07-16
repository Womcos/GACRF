import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

__all__ = ['LabelSmoothing', 'NLLMultiLabelSmooth', 'SegmentationLosses']

def get_batch_label_vector(target, nclass):
    # target is a 3D Variable BxHxW, output is 2D BxnClass
    batch = target.size(0)
    tvect = Variable(torch.zeros(batch, nclass))
    for i in range(batch):
        hist = torch.histc(target[i].cpu().data.float(),
                           bins=nclass, min=0,
                           max=nclass - 1)
        vect = hist > 0
        tvect[i] = vect
    return tvect

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class NLLMultiLabelSmooth(nn.Module):
    def __init__(self, smoothing = 0.1):
        super(NLLMultiLabelSmooth, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim = -1)
    
            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
    
            smooth_loss = -logprobs.mean(dim=-1)
    
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
    
            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)

class CrossEntropyLoss(nn.Module):
    def __init__(self, OHEM=False, ignore_index=-1):
        super(CrossEntropyLoss, self).__init__()
        self.OHEM = OHEM
        self.ignore_index = ignore_index
        if OHEM:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116,
                 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
            self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
            self.thresh = 0.7
            self.min_kept = 100000
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, predict, target, do_OHEM=False):
        if self.OHEM:
            """
                        Args:
                            predict:(n, c, h, w)
                            target:(n, h, w)
                            weight (Tensor, optional): a manual rescaling weight given to each class.
                                                       If given, has to be a Tensor of size "nclasses"
                    """
            if do_OHEM:
                assert not target.requires_grad
                assert predict.dim() == 4
                assert target.dim() == 3
                assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
                assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
                assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

                n, c, h, w = predict.size()
                input_label = target.data.cpu().numpy().ravel().astype(np.int32)
                x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
                input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
                input_prob /= input_prob.sum(axis=0).reshape((1, -1))

                valid_flag = input_label != self.ignore_index
                valid_inds = np.where(valid_flag)[0]
                label = input_label[valid_flag]
                num_valid = valid_flag.sum()
                if self.min_kept >= num_valid:
                    print('Labels: {}'.format(num_valid))
                elif num_valid > 0:
                    prob = input_prob[:, valid_flag]
                    pred = prob[label, np.arange(len(label), dtype=np.int32)]
                    threshold = self.thresh
                    if self.min_kept > 0:
                        index = pred.argsort()
                        threshold_index = index[min(len(index), self.min_kept) - 1]
                        if pred[threshold_index] > self.thresh:
                            threshold = pred[threshold_index]
                    kept_flag = pred <= threshold
                    valid_inds = valid_inds[kept_flag]
                    # print('hard ratio: {} = {} / {} '.format(round(len(valid_inds)/num_valid, 4), len(valid_inds), num_valid))

                label = input_label[valid_inds].copy()
                input_label.fill(self.ignore_index)
                input_label[valid_inds] = label
                valid_flag_new = input_label != self.ignore_index
                # print(np.sum(valid_flag_new))
                target = Variable(torch.from_numpy(input_label.reshape(target.size())).long().cuda())

                return self.criterion(predict, target)
            else:
                return self.criterion(predict, target)
        else:
            return self.criterion(predict, target)

class OneLoss(nn.Module):
    def __init__(self, OHEM, ignore_index):
        super(OneLoss, self).__init__()
        self.ignore_index = ignore_index
        self.seg_criterion = CrossEntropyLoss(OHEM, ignore_index)
        self.reg_criterion = nn.BCELoss()

    def forward(self, predict, target, do_OHEM=False):
        if predict.dim() == 4:
            return self.seg_criterion.forward(predict, target, do_OHEM)
        elif predict.dim() == 2:
            b, nclass = predict.shape
            target = get_batch_label_vector(target, nclass=nclass).type_as(predict)   # b, nclass
            loss = self.reg_criterion.forward(torch.sigmoid(predict), target)
            return loss
        else:
            raise Exception('Wrong output size.')

class SegmentationLosses(nn.Module):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, DS=False, CTF=False, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.4, weight=None,ignore_index=-1, OHEM=False):
        super(SegmentationLosses, self).__init__()
        self.criterion = OneLoss(OHEM, ignore_index)
        self.DS = DS
        self.CTF = CTF
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight) 

    def forward(self, *inputs):
        if self.CTF:
            logits_num = len(tuple(inputs)) - 1  # target occluded
            preds = tuple(inputs)[:-1]
            target = tuple(inputs)[-1]
            
            loss = self.criterion.forward(preds[0], target, do_OHEM=True)
            for i in range(1, logits_num):
                loss = loss + self.criterion.forward(preds[i], target)
            return loss
        if self.DS:
            logits_num = len(tuple(inputs)) - 1  # target occluded
            weights = np.array([0.5 ** i for i in range(logits_num)])
            #weights = weights / weights.sum()
            preds = tuple(inputs)[:-1]
            target = tuple(inputs)[-1]
            assert len(weights) == len(preds)
            loss = weights[0] * self.criterion.forward(preds[0], target, do_OHEM=True)
            for i in range(1, logits_num):
                loss = loss + weights[i] * self.criterion.forward(preds[i], target)
            return loss
        if not self.se_loss and not self.aux:
            return self.criterion.forward(*inputs)
        elif not self.se_loss:
            logits_num = len(tuple(inputs)) - 1  # target occluded
            preds = tuple(inputs)[:-1]
            target = tuple(inputs)[-1]
            loss = self.criterion.forward(preds[0], target)
            for i in range(1, logits_num):
                loss = loss + self.aux_weight * self.criterion.forward(preds[i], target)
            return loss
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = self.criterion.forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = self.criterion.forward(pred1, target)
            loss2 = self.criterion.forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect
