# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2022/5/17 22:45
# @brief      : 模型训练代码
"""

import os
import sys

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

import pandas as pd
import pytorch_lightning as pl

from sklearn.model_selection import StratifiedGroupKFold
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from config.uw_config import cfg
from datasets.uw_madison_gi import UWDataModule
from model.uw_model import UWModel

if __name__ == "__main__":
    pl.seed_everything(cfg.seed)

    wandb_logger = WandbLogger(project="UW-Madison-GI-Tract-Image-Segmentation", config=cfg, group='cv',
                               job_type='train', anonymous=False)

    df = pd.read_csv(f'{cfg.DATASET_DIR}-mask/train.csv')
    df['segmentation'] = df['segmentation'].fillna('')
    df['rle_len'] = df['segmentation'].map(len)

    df2 = df.groupby(['id'])['segmentation'].agg(list).to_frame().reset_index()
    df2 = df2.merge(df.groupby(['id'])['rle_len'].agg(sum).to_frame().reset_index())
    df = df.drop(columns=['segmentation', 'class', 'rle_len'])
    df = df.groupby(['id']).head(1).reset_index(drop=True)

    df = df.merge(df2, on=['id'])
    df['empty'] = (df['rle_len'] == 0)

    fault1 = 'case7_day0'
    fault2 = 'case81_day30'
    df = df[~df['id'].str.contains(fault1) & ~df['id'].str.contains(fault2)].reset_index(drop=True)

    skf = StratifiedGroupKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['empty'], groups=df['case'])):
        df.loc[val_idx, 'fold'] = fold

    for fold in range(cfg.n_fold):
        print(f'Start Train Fold: {fold}', flush=True)

        model_checkpoint = ModelCheckpoint(
            dirpath=cfg.output_path,
            filename=f'{fold}-' + '{epoch}-{val_dice:4f}',
            save_top_k=1,
            verbose=True,
            monitor='val_dice',
            mode='max'
        )

        early_stopping_callback = EarlyStopping(
            mode='max',
            monitor='val_dice',
            patience=cfg.patience
        )

        trainer = pl.Trainer(
            logger=wandb_logger,
            callbacks=[model_checkpoint, early_stopping_callback],
            num_sanity_val_steps=0,
            accumulate_grad_batches=1,
            max_epochs=cfg.T_max,
            check_val_every_n_epoch=1,
            gpus=-1,
            progress_bar_refresh_rate=15,
            precision=16
        )

        data_module = UWDataModule(df, fold)
        model = UWModel(arch='Unet', encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=3)

        trainer.fit(model, data_module)
