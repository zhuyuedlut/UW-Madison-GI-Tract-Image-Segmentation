# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2022/6/3 21:50
# @brief      : 常用工具函数定义
"""

import pandas as pd

from config.uw_config import cfg


def preprocess_data():
    train_df = pd.read_csv(f'{cfg.DATASET_DIR}/train.csv')
    train_df['segmentation'] = train_df['segmentation'].fillna('')
    train_df['rle_len'] = train_df['segmentation'].map(len)

    train_df['mask_path'] = train_df.mask_path.str.replace('/png/', '/np').str.replace('.png', '.npy')
    temp_df = train_df.groupby(['id'])['segmentation'].agg(list).to_frame().reset_index()
    temp_df = temp_df.merge(train_df.groupby(['id'])['rle_len'].agg(sum).to_frame().reset_index())

    train_df = train_df.drop(columns=['segmentation', 'class', 'rle_len'])
    train_df = train_df.groupby(['id']).head(1).reset_index(drop=True)
    train_df = train_df.merge(temp_df, on=['id'])
    train_df['empty'] = (train_df.rle_len == 0)

    fault1 = 'case7_day0'
    fault2 = 'case81_day30'
    train_df = train_df[~train_df['id'].str.contains(fault1) & ~train_df['id'].str.contains(fault2)].reset_index(
        drop=True)

    return train_df


if __name__ == "__main__":
    df = preprocess_data()
    print(df.head())