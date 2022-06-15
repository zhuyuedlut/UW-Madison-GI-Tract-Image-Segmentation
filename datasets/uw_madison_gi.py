# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2022/5/14 23:26
# @brief      :  uw-madison-gi数据集读取
"""
import os.path

import cv2
import numpy as np
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader

from config.uw_config import cfg
from tools.utils import collate_fn


class UWDataset(Dataset):
    def __init__(self, df, transforms=None):
        super(UWDataset, self).__init__()
        self.df = df
        self.ids = df['id'].tolist()

        self.img_paths = df['image_path'].tolist()
        self.mask_paths = df['mask_path'].tolist()

        self.transforms = transforms
        self.n_25d_shift = cfg.n_25d_shift

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        if cfg.use_25d:
            img = self.load_2_5d_slice(img_path)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = np.tile(img[..., None], [1, 1, 3]).astype('float32')
            mx = np.max(img)
            if mx:
                img /= mx  # scale image to [0, 1]

        msk_path = self.mask_paths[index]
        msk = self.load_2_5d_slice(msk_path)

        ### augmentations
        data = self.transforms(image=img, mask=msk)
        img = data['image']
        msk = data['mask']
        img = np.transpose(img, (2, 0, 1))  # [c, h, w]
        msk = np.transpose(msk, (2, 0, 1))  # [c, h, w]
        return torch.tensor(img), torch.tensor(msk)

    def load_2_5d_slice(self, middle_img_path):
        #### 步骤1: 获取中间图片的基本信息
        #### eg: middle_img_path: 'slice_0005_266_266_1.50_1.50.png'
        middle_slice_num = os.path.basename(middle_img_path).split('_')[1]  # eg: 0005
        middle_str = 'slice_' + middle_slice_num
        middle_img = cv2.imread(middle_img_path, cv2.IMREAD_UNCHANGED)

        new_25d_imgs = []

        ##### 步骤2：按照左右n_25d_shift数量进行填充，如果没有相应图片填充为Nan.
        ##### 注：经过EDA发现同一天的所有患者图片的shape是一致的
        for i in range(-self.n_25d_shift, self.n_25d_shift + 1):  # eg: i = {-2, -1, 0, 1, 2}
            shift_slice_num = int(middle_slice_num) + i
            shift_str = 'slice_' + str(shift_slice_num).zfill(4)
            shift_img_path = middle_img_path.replace(middle_str, shift_str)

            if os.path.exists(shift_img_path):
                shift_img = cv2.imread(shift_img_path, cv2.IMREAD_UNCHANGED)  # [w, h]
                new_25d_imgs.append(shift_img)
            else:
                new_25d_imgs.append(None)

        ##### 步骤3：从中心开始往外循环，依次填补None的值
        ##### eg: n_25d_shift = 2, 那么形成5个channel, idx为[0, 1, 2, 3, 4], 所以依次处理的idx为[1, 3, 0, 4]
        shift_left_idxs = []
        shift_right_idxs = []
        for related_idx in range(self.n_25d_shift):
            shift_left_idxs.append(self.n_25d_shift - related_idx - 1)
            shift_right_idxs.append(self.n_25d_shift + related_idx + 1)

        for left_idx, right_idx in zip(shift_left_idxs, shift_right_idxs):
            if new_25d_imgs[left_idx] is None:
                new_25d_imgs[left_idx] = new_25d_imgs[left_idx + 1]
            if new_25d_imgs[right_idx] is None:
                new_25d_imgs[right_idx] = new_25d_imgs[right_idx - 1]

        new_25d_imgs = np.stack(new_25d_imgs, axis=2).astype('float32')  # [w, h, c]
        mx_pixel = new_25d_imgs.max()
        if mx_pixel != 0:
            new_25d_imgs /= mx_pixel
        return new_25d_imgs


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
                          num_workers=cfg.workers, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=cfg.valid_bs, shuffle=False, drop_last=True,
                          num_workers=cfg.workers)
