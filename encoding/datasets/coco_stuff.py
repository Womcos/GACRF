###########################################################################
# Created by: Xiaofeng Ding
# Email: dxfeng@shu.edu.cn
# Copyright (c) 2020
###########################################################################

import os
import random
import scipy.io
import numpy as np
from PIL import Image, ImageOps, ImageFilter

import torch
from .base import BaseDataset
from ..path import root_data

class COCO_stuff_segmentation(BaseDataset):
    NUM_CLASS = 171
    BASE_DIR = 'COCO_stuff'
    def __init__(self, root=root_data, split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(COCO_stuff_segmentation, self).__init__(root, split, mode, transform,
                                                 target_transform, **kwargs)
        _root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(_root, 'annotations_png')
        _image_dir = os.path.join(_root, 'images')
        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_root, 'imageLists')
        if self.mode == 'train':
            _split_f = os.path.join(_splits_dir, 'train.txt')
        elif self.mode == 'val':
            _split_f = os.path.join(_splits_dir, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split.')
        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n')+".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                if self.mode != 'test':
                    _mask = os.path.join(_mask_dir, line.rstrip('\n')+".png")
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                _img = self.transform(_img)
            return _img, os.path.basename(self.images[index])
        _target = Image.open(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            _img, _target = self._sync_transform( _img, _target)
        elif self.mode == 'val':
            _img, _target = self._val_sync_transform( _img, _target)
        # general resize, normalize and toTensor
        if self.transform is not None:
            _img = self.transform(_img)
        if self.target_transform is not None:
            _target = self.target_transform(_target)
        return _img, _target

    def __len__(self):
        return len(self.images)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32') - 2
        target[target == -2] = -1
        return torch.from_numpy(target).long()
