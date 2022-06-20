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
import segmentation_models_pytorch as smp
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

    JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
    DiceLoss = smp.losses.DiceLoss(mode='multilabel')
    BCELoss = smp.losses.SoftBCEWithLogitsLoss()
    LovaszLoss = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
    TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)

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

    def loss_fn(y_pred, y_true):
        return 0.5 * BCELoss(y_pred, y_true) + 0.5 * TverskyLoss(y_pred, y_true)


    for fold in range(cfg.n_fold):
        print(f'Start Train Fold: {fold}', flush=True)

        model_checkpoint = ModelCheckpoint(
            dirpath=cfg.output_path,
            filename=f'{fold}-' + '{epoch}-{val_loss:2f}',
            save_top_k=1,
            verbose=True,
            monitor='val_loss',
            mode='min'
        )

        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=cfg.patience
        )

        trainer = pl.Trainer(
            logger=wandb_logger,
            callbacks=[model_checkpoint, early_stopping_callback],
            num_sanity_val_steps=0,
            accumulate_grad_batches=2,
            max_epochs=cfg.T_max,
            gpus=-1,
            progress_bar_refresh_rate=15,
            precision=16
        )

        data_module = UWDataModule(df, fold)
        model = UWModel(arch='Unet', encoder_name='efficientnet-b6', encoder_weights='imagenet', in_channels=3, classes=3, loss_fn=loss_fn)
        trainer.fit(model, data_module)