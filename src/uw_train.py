# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2022/5/17 22:45
# @brief      : 模型训练代码
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from config.uw_config import cfg
from datasets.uw_madison_gi import UWDataModule
from model.uw_model import UWModel
from tools.callbacks import model_checkpoint, early_stopping_callback

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

    df = pd.read_csv('../data/train.csv')
    df['segmentation'] = df['segmentation'].astype('str')

    fractions = np.array([0.8, 0.2])
    df_train, df_val = np.array_split(df, (fractions[:-1].cumsum() * len(df)).astype(int))

    data_module = UWDataModule(df_train, df_val)




