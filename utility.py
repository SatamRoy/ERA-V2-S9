#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function
import torch
import torchvision


from albumentations.pytorch import ToTensorV2


#  Custom Data Transformer
import albumentations as A
import numpy as np

class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.transform = transform
        self.cifar10_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

    def __len__(self):
        return len(self.cifar10_data)

    def __getitem__(self, idx):
        image, label = self.cifar10_data[idx]
        if self.transform:
            image = self.transform(image=np.array(image))["image"]
        return image, label



train_transforms = A.Compose([
    A.HorizontalFlip(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
    A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=[0.5, 0.5, 0.5], mask_fill_value=None, p=0.5),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2(),
])



test_transforms = A.Compose([
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2(),
])

