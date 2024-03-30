# ERA-V2-S9
Here we are building a CNN Image classification model for CIFAR10 dataset.

- model.py:- Our CNN Model with the following criteria:

4 convolution blocks in total , no max-pooling, dilated convolutions,
Receptive field of 49
GAP Layer
Target accuracy >85% (achieved 87%)
utility.py : - Use of albumentation library for data augmentation with horizontal flip, shiftScaleRotate and coarseDropout
