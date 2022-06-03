# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2022/5/14 23:26
# @brief      :  uw-madison-gi数据集读取
"""
import numpy as np
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, ToTensor

from PIL import Image

from config.uw_config import cfg


def load_image(path):
    image = Image.open(path).convert('RGB')
    return image


class UWDataset(Dataset):
    def __init__(self, df, transforms=None):
        super(UWDataset, self).__init__()
        self.df = df
        self.ids = df['ids'].tolist()

        if 'mask_path' in df.columns:
            self.img_paths = df['image_path'].tolist()
            self.mask_paths = df['mask_path'].tolist()
        else:
            self.img_paths = df['image_paths'].tolist()

        self.resize = Resize((cfg.height, cfg.width))

        self.transforms = transforms


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        image = load_image(img_path)
        mask_path = self.mask_paths[index]
        mask = load_image(mask_path)

        image = ToTensor()(self.resize(image))
        mask = ToTensor()(self.resize(mask))

        if self.transforms is not None:
            data = self.transforms(image=image, mask=mask)
            image, mask = data['image'], data['mask']

        return image, mask


class UWDataModule(pl.LightningDataModule):
    def __init__(self, df, fold):
        super(UWDataModule, self).__init__()
        self.df = df
        self.fold = fold

        self.train_df = df.query("fold!=@fold").reset_index(drop=True)
        self.valid_df = df.query("fold==@fold").reset_index(drop=True)

        self.train_dataset = None
        self.valid_dataset = None

    def setup(self, stage=None):
        self.train_dataset = UWDataset(self.train_df)
        self.valid_dataset = UWDataset(self.valid_df)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=cfg.train_bs, shuffle=True, drop_last=True,
                          num_workers=cfg.workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=cfg.valid_bs,drop_last=True, num_workers=cfg.workers)


