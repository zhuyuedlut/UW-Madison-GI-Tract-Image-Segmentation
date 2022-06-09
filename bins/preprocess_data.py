# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2022/6/8 20:18
# @brief      : 数据预处理
"""
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from glob import glob
from joblib import Parallel, delayed
from tqdm import tqdm
tqdm.pandas()

from config.uw_config import cfg
from tools.utils import get_metadata, path2info, rle_decode, load_img, show_img

if __name__ == "__main__":
    df = pd.read_csv(f'{cfg.DATASET_DIR}/train.csv')
    df = df.progress_apply(get_metadata, axis=1)

    paths = glob(f'{cfg.DATASET_DIR}/train/*/*/*/*')
    path_df = pd.DataFrame(paths, columns=['image_path'])
    path_df = path_df.apply(path2info, axis=1)

    df = df.merge(path_df, on=['case', 'day', 'slice'])


    def id2mask(slice_id):
        idf = df[df['id'] == slice_id]
        wh = idf[['height', 'width']].iloc[0]
        shape = (wh.height, wh.width, 3)
        mask = np.zeros(shape, dtype=np.uint8)
        for i, cls in enumerate(['large_bowel', 'small_bowel', 'stomach']):
            cdf = idf[idf['class'] == cls]
            rle = cdf.segmentation.squeeze()
            if len(cdf) and not pd.isna(rle):
                mask[..., i] = rle_decode(rle, shape[:2])
        return mask


    def save_mask(slice_id):
        idf = df[df['id'] == slice_id]
        mask = id2mask(slice_id)
        image_path = idf.image_path.iloc[0]
        mask_path = image_path.replace('uw-madison-gi-tract-image-segmentation',
                                       'uw-madison-gi-tract-image-segmentation-mask')
        mask_folder = mask_path.rsplit('/', 1)[0]
        os.makedirs(mask_folder, exist_ok=True)
        cv2.imwrite(mask_path, mask * 255, [cv2.IMWRITE_PNG_COMPRESSION, 1])


    ids = df['id'].unique()
    columns = ["id", "case", "day", "slice", "empty", "image"]
    class_labels = {
        1: "Large Bowel",
        2: "Small Bowel",
        3: "Stomach",
    }

    _ = Parallel(n_jobs=4, backend='threading')(delayed(save_mask)(slice_id) for slice_id in tqdm(ids, total=len(ids)))
    df['mask_path'] = df.image_path.str.replace('uw-madison-gi-tract-image-segmentation',
                                                'uw-madison-gi-tract-image-segmentation-mask')

    DIR = cfg.DATASET_DIR.replace('uw-madison-gi-tract-image-segmentation',
                                  'uw-madison-gi-tract-image-segmentation-mask')
    df.to_csv(f'{DIR}/train.csv', index=False)

    # test show image
    i = 250
    img = load_img(df.image_path.iloc[i])
    mask_path = df['image_path'].iloc[i].replace('uw-madison-gi-tract-image-segmentation',
                                                 'uw-madison-gi-tract-image-segmentation-mask')
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    plt.figure(figsize=(5, 5))
    show_img(img, mask=mask)
    plt.show()


