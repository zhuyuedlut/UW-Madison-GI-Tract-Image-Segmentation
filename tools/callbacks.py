# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2022/5/17 22:51
# @brief      : pytorch-lightning train中的callback
"""

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from config.uw_config import cfg

model_checkpoint = ModelCheckpoint(
    dirpath=cfg.output_path,
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min'
)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=cfg.patience
)
