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

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import pytorch_lightning as pl

from sklearn.model_selection import StratifiedGroupKFold
from pytorch_lightning.loggers import WandbLogger

from config.uw_config import cfg
from datasets.uw_madison_gi import UWDataModule
from model.uw_model import UWModel
from tools.callbacks import model_checkpoint, early_stopping_callback

from tools.utils import preprocess_data

if __name__ == "__main__":
    pl.seed_everything(cfg.seed)

    wandb_logger = WandbLogger(project="UW-Madison-GI-Tract-Image-Segmentation", config=cfg, group='cv',
                               job_type='train', anonymous=False)

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[model_checkpoint, early_stopping_callback],
        num_sanity_val_steps=0,
        max_epochs=cfg.T_max,
        gpus=-1,
        progress_bar_refresh_rate=15,
        precision=16
    )

    JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
    DiceLoss = smp.losses.DiceLoss(mode='multilabel')
    BCELoss = smp.losses.SoftBCEWithLogitsLoss()
    LovaszLoss = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
    TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)


    def loss_fn(y_pred, y_true):
        return 0.5 * BCELoss(y_pred, y_true) + 0.5 * TverskyLoss(y_pred, y_true)

    df = preprocess_data()
    skf = StratifiedGroupKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['empty'], groups=df['case'])):
        df.loc[val_idx, 'fold'] = fold

    for fold in range(cfg.n_fold):
        print(f'Start Train Fold: {fold}', flush=True)
        data_module = UWDataModule(df, fold)
        model = UWModel(arch='Unet', encoder_name='resnet34', in_channels=3, classes=3, loss_fn=loss_fn)
        trainer.fit(model, data_module)