# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2022/6/9 22:57
# @brief      : 将训练数据按照时间序列重新组合成训练图片（2.5D）
"""

import cv2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.patches import Rectangle
from tqdm.notebook import tqdm

tqdm.pandas()
from joblib import Parallel, delayed

if __name__ == "__main__":
    IMG_SIZE = [320, 384]
    DATASET_DIR = r'/home/zhuyuedlut/Datasets/uw-madison-gi-tract-image-segmentation-mask'
    SAVE_DATASET_DIR = r'/home/zhuyuedlut/Datasets/uw-madison-gi-tract-image-segmentation-2.5d'


    def load_img(path, size=IMG_SIZE):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        shape0 = np.array(img.shape[:2])
        resize = np.array(size)

        if np.any(shape0 != resize):
            diff = resize - shape0
            pad0 = diff[0]
            pad1 = diff[1]
            pady = [pad0 // 2, pad0 // 2 + pad0 % 2]
            padx = [pad1 // 2, pad1 // 2 + pad1 % 2]

            img = np.pad(img, [pady, padx])
            img = img.reshape(*resize)

        return img


    def load_msk(path, size=IMG_SIZE):
        msk = cv2.imread(path, cv2.COLOR_BGR2RGB)
        shape0 = np.array(msk.shape[:2])
        resize = np.array(size)
        if np.any(shape0 != resize):
            diff = resize - shape0
            pad0 = diff[0]
            pad1 = diff[1]
            pady = [pad0 // 2, pad0 // 2 + pad0 % 2]
            padx = [pad1 // 2, pad1 // 2 + pad1 % 2]
            msk = np.pad(msk, [pady, padx, [0, 0]])
            msk = msk.reshape((*resize, 3))

        return msk


    def show_img(img, mask=None):
        plt.imshow(img, cmap='bone')

        if mask is not None:
            plt.imshow(mask, alpha=0.5)
            handles = [Rectangle((0, 0), 1, 1, color=_c) for _c in
                       [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]]
            labels = ["Large Bowel", "Small Bowel", "Stomach"]
            plt.legend(handles, labels)
        plt.axis('off')


    def load_imgs(img_paths, size=IMG_SIZE):
        imgs = np.zeros((*size, len(img_paths)), dtype=np.uint16)
        for i, img_path in enumerate(img_paths):
            img = load_img(img_path, size=size)
            imgs[..., i] += img
        return imgs


    df = pd.read_csv(f'{DATASET_DIR}/train.csv')
    df['segmentation'] = df['segmentation'].fillna('')
    df['rle_len'] = df['segmentation'].map(len)
    df2 = df.groupby(['id'])['segmentation'].agg(list).to_frame().reset_index()
    df2 = df2.merge(df.groupby(['id'])['rle_len'].agg(sum).to_frame().reset_index())

    df = df.drop(columns=['segmentation', 'class', 'rle_len'])
    df = df.groupby(['id']).head(1).reset_index(drop=True)
    df = df.merge(df2, on=['id'])
    df['empty'] = (df.rle_len == 0)

    channels = 3
    stride = 2
    for i in range(channels):
        df[f'image_path_{i:02}'] = df.groupby(['case', 'day'])['image_path'].shift(-i * stride).fillna(method="ffill")
    df['image_paths'] = df[[f'image_path_{i:02d}' for i in range(channels)]].values.tolist()


    def save_mask(id_):
        row = df[df['id'] == id_].squeeze()

        img_paths = row.image_paths
        imgs = load_imgs(img_paths)
        np.save(f'{SAVE_DATASET_DIR}/imgs/{id_}.npy', imgs)

        msk_path = row.mask_path
        msk = load_msk(msk_path)
        np.save(f'{SAVE_DATASET_DIR}/masks/{id_}.npy', msk)

        return


    ids = df['id'].unique()
    _ = Parallel(n_jobs=-1, backend='threading')(delayed(save_mask)(id_) for id_ in tqdm(ids, total=len(ids)))

    df.to_csv(f'{SAVE_DATASET_DIR}/train.csv', index=False)