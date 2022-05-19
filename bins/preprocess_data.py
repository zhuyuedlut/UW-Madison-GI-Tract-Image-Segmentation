# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2022/5/14 21:30
# @brief      : UW-Madison-GI数据预处理看
"""

import os

import pandas as pd

from glob import glob


def fetch_file_from_id(root_dir, case_id):
    case_folder = case_id.split('_')[0]
    day_folder = '_'.join(case_id.split('_')[:2])
    file_starter = '_'.join(case_id.split('_')[2:])

    folder = os.path.join(root_dir, case_folder, day_folder, 'scans')
    file = glob(f"{folder}/{file_starter}*")

    return file[0]


if __name__ == '__main__':
    DATASET_DIR = r'/mnt/datasets/uw-madison-gi-tract-image-segmentation'
    train_df = pd.read_csv(f'{DATASET_DIR}/train.csv')

    train_df['segmentation'] = train_df['segmentation'].astype('str')
    train_df['case_id'] = train_df['id'].apply(lambda x: x.split('_')[0][4:])
    train_df['day_id'] = train_df['id'].apply(lambda x: x.split('_')[1][3:])
    train_df['slice_id'] = train_df['id'].apply(lambda x: x.split('_')[-1])

    train_df["path"] = train_df["id"].apply(lambda x: fetch_file_from_id(f'{DATASET_DIR}/train', x))

    train_df["height"] = train_df["path"].apply(lambda x: os.path.split(x)[-1].split("_")[2]).astype("int")
    train_df["width"] = train_df["path"].apply(lambda x: os.path.split(x)[-1].split("_")[3]).astype("int")

    class_names = train_df["class"].unique()
    for index, label in enumerate(class_names):
        # replacing class names with indexes
        train_df["class"].replace(label, index, inplace=True)

    train_df.to_csv(os.path.join(os.path.abspath('.'), 'data/train.csv'))