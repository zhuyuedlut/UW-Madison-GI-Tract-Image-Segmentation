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


def prepare_mask_data(segmentation):
    all_values = map(int, segmentation.split(' '))
    starterIndex, pixCount = [], []

    for index, value in enumerate(all_values):
        if index % 2:
            pixCount.append(value)
        else:
            starterIndex.append(value)

    return starterIndex, pixCount


def fetch_pos_pixel_indexes(indexes, counts):
    final_arr = []
    for index, counts in zip(indexes, counts):
        final_arr += [index + i for i in range(counts)]

    return final_arr


def prepare_mask(segmentation, height, width):
    indexes, counts = prepare_mask_data(segmentation)
    pos_pixel_indexes = fetch_pos_pixel_indexes(indexes, counts)
    mask_array = np.zeros(height * width)
    mask_array[pos_pixel_indexes] = 1

    return mask_array


class UWDataset(Dataset):
    def __init__(self, df, width=256, height=256, transforms=None):
        super(UWDataset, self).__init__()
        self.df = df
        self.width = width
        self.height = height
        self.resize = Resize((width, height))
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        path = self.df.loc[index, "path"]
        image = load_image(path)
        mask_h, mask_w = self.df.loc[index, 'height'], self.df.loc[index, 'width']
        segmentation = self.df.loc[index, 'segmentation']
        label = self.load_mask(segmentation, height=mask_h, width=mask_w)

        image = ToTensor()(self.resize(image))
        label = ToTensor()(self.resize(label))

        mask = torch.zeros((3, self.height, self.width))
        class_label = self.df.loc[index, 'class']
        mask[class_label, ...] = label

        if self.transforms is not None:
            image = self.transforms(image)
            label = self.transforms(label)
            return ToTensor()(image), ToTensor()(label)
        else:
            return image, label

    def load_mask(self, segmentation, height, width):
        if segmentation != 'nan':
            return Image.fromarray(prepare_mask(segmentation, height, width))
        return Image.fromarray(np.zeros((height, width)))


class UWDataModule(pl.LightningDataModule):
    def __init__(self, df_train, df_valid):
        super(UWDataModule, self).__init__()
        self.df_train = df_train
        self.df_valid = df_valid

        self.train_dataset = None
        self.valid_dataset = None

    def setup(self, stage=None):
        self.train_dataset = UWDataset(self.df_train)
        self.valid_dataset = UWDataset(self.df_valid)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=cfg.train_bs, shuffle=True, drop_last=True,
                          num_workers=cfg.workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=cfg.valid_bs, shuffle=True, drop_last=True,
                          num_workers=cfg.workers)


if __name__ == "__main__":
    import pandas as pd


    df = pd.read_csv('../data/train.csv')
    df['segmentation'] = df['segmentation'].astype('str')

    dataset = UWDataset(df)
    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

    for i, data in enumerate(train_dataloader):
        print(data)

