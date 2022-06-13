# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2022/6/3 21:50
# @brief      : 常用工具函数定义
"""
import cv2
import numpy as np

import matplotlib.pyplot as plt
import torch

from matplotlib.patches import Rectangle


def get_metadata(row):
    """
    从train.csv中id列中解析处case day slice
    :param row:
    :return:
    """
    case, day, _, slice_id = row['id'].split('_')
    case, day = case.replace('case', ''), day.replace('day', '')
    row['case'], row['day'], row['slice'] = int(case), int(day), int(slice_id)

    return row


def path2info(row):
    """
    根据image_path中的训练图片的名称解析处height, width, case, day, slice
    :param row:
    :return:
    """
    path = row['image_path']
    data = path.split('/')
    slice_id = int(data[-1].split('_')[1])
    case = int(data[-3].split('_')[0].replace('case',''))
    day = int(data[-3].split('_')[1].replace('day',''))
    width = int(data[-1].split('_')[2])
    height = int(data[-1].split('_')[3])

    row['height'] = height
    row['width'] = width
    row['case'] = case
    row['day'] = day
    row['slice'] = slice_id

    return row


def rle_decode(mask_rle, shape):
    """
    将rle字符串还原为对应的H * W的矩阵（单通道图片）
    :param mask_rle:
    :param shape:
    :return:
    """
    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.astype('float32')
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype('uint8')
    return img


def show_img(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    plt.imshow(img, cmap='bone')

    if mask is not None:
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0,0),1,1, color=_c) for _c in [(0.667,0.0,0.0), (0.0,0.667,0.0), (0.0,0.0,0.667)]]
        labels = [ "Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles,labels)
    plt.axis('off')


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def collate_fn(batch):
    batch = list(zip(*batch))
    img_list = batch[0]
    mask_list = batch[1]
    max_size = _max_by_axis([list(img.shape) for img in img_list])
    batch_shape = [len(img_list)] + max_size
    b, c, h, w = batch_shape
    dtype = img_list[0].dtype
    device = img_list[0].device
    pad_imgs = torch.zeros(batch_shape, dtype=dtype, device=device)
    pad_masks = torch.zeros(batch_shape, dtype=dtype, device=device)
    for img, mask, pad_img, pad_mask in zip(img_list, mask_list, pad_imgs, pad_masks):
        # img/pad_img:[c, w, h]
        pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
        pad_mask[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(mask)
    return [pad_imgs, pad_masks]


