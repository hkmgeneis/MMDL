# -*- coding:utf-8 -*-
"""
   Custom dataset that includes image file names.
   Extends torchvision.datasets.ImageFolder

   @author: xisx
   @time: 20201124
"""
from torchvision import datasets


class ImageFolderWithPaths(datasets.ImageFolder):
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        img = path.split('/')[-1]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (img,))
        return tuple_with_path
