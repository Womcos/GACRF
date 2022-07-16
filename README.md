# GACRF
## Introduction

We propose a novel Conditional Random Field (CRF) based network to adaptively integrate the semantic contextual information, i.e. group-wise adaptive CRF (GA-CRF) network. The GA-CRF network shrinks the semantic gap between different scales and enhances the model structure reasoning. Extensive experiments on the PASCAL VOC 2012, Cityscapes, and PASCAL Context datasets demonstrate that our method significantly improves the segmentation results and achieves superior performance.

## Usage

1. Install pytorch

   Our code is conducted on python 3.6 and torch 1.8.0.

2. Install environment

   clone the repository and open the folder, run

   ```
   python setup.py
   ```

   or

   ```
   pip install -e .
   ```

3. Dataset

   **PASCAL VOC 2012**: Download the [PASCAL VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)  and [augmentation data](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0), then convert the dataset into trainaug, trainval, and test sets for training, fine-tuning and testing, respectively.

   **Cityscapes**: Download the [Cityscapes dataset](https://www.cityscapes-dataset.com/) and convert the dataset to [19 categories](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py).

   **PASCAL Context**: run script `scripts/prepare_pcontext.py`.

4. Training on PASCAL VOC 2012 dataset

   ```
   cd ./experiments/segmentation/
   ```

   Training on the trainaug set:

   ```
   CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset pascal_aug2 --backbone resnet101s --model gacrf --checkname myname --ft --DS --dilated
   ```

   Fine-tuning on the trainval set:

   ```
   CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset pascal_voc --backbone resnet101s --model gacrf --checkname myname --resume "pretrained_model_path" --ft --DS --dilated
   ```

   where `pretrained_model_path` is the path to the model trained on trainaug set.

5. Evaluation on PASCAL VOC 2012 dataset

   ```
   cd ./experiments/segmentation/
   ```

   Single scale testing on val set for model (trained on trainaug set):

   ```
   CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset pascal_voc --backbone resnet101s --model gacrf --resume "pretrained_model_path" --eval --dilated
   ```



