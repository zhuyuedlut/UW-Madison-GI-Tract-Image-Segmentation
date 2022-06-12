# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2022/5/14 23:26
# @brief      :  uw-madison-gi数据集读取
"""
import cv2
import numpy as np
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader

from PIL import Image

from config.uw_config import cfg


def load_image(path):
    image = Image.open(path).convert('RGB')
    return image


class UWDataset(Dataset):
    def __init__(self, df, transforms=None):
        super(UWDataset, self).__init__()
        self.df = df
        self.ids = df['id'].tolist()

        self.img_paths = df['image_path'].tolist()
        self.mask_paths = df['mask_path'].tolist()

        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        if cfg.use_25d:
           img = np.load(img_path).astype('float32')
        else:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = np.tile(img[..., None], [1, 1, 3]).astype('float32')

        mx = np.max(img)
        if mx:
            img /= mx  # scale image to [0, 1]

        msk_path = self.mask_paths[index]
        if cfg.use_25d:
            msk = np.load(msk_path).astype('float32')
        else:
            msk = cv2.imread(msk_path, cv2.COLOR_BGR2RGB).astype('float32')
            msk /= 255.0  # scale mask to [0, 1]

        ### augmentations
        data = self.transforms(image=img, mask=msk)
        img = data['image']
        msk = data['mask']
        img = np.transpose(img, (2, 0, 1))  # [c, h, w]
        msk = np.transpose(msk, (2, 0, 1))  # [c, h, w]
        return torch.tensor(img), torch.tensor(msk)


class UWDataModule(pl.LightningDataModule):
    def __init__(self, df, fold):
        super(UWDataModule, self).__init__()
        self.df = df
        self.fold = fold
        self.transforms = cfg.data_transforms

        self.train_df = df.query("fold!=@fold").reset_index(drop=True)
        self.valid_df = df.query("fold==@fold").reset_index(drop=True)

        self.train_dataset = None
        self.valid_dataset = None

    def setup(self, stage=None):
        self.train_dataset = UWDataset(self.train_df, transforms=self.transforms['train'])
        self.valid_dataset = UWDataset(self.valid_df, transforms=self.transforms['valid_test'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=cfg.train_bs, shuffle=True, drop_last=True,
                          num_workers=cfg.workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=cfg.valid_bs, drop_last=True, num_workers=cfg.workers)
