# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2022/5/15 22:16
# @brief      : uw segmentation不同的model实现
"""
import torch
import torch.optim as optim
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from torch.optim import lr_scheduler
from config.uw_config import cfg


class UWModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, encoder_weights, classes, in_channels, loss_fn):
        super(UWModel, self).__init__()
        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )

        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer('std', torch.tensor(params['std']).view(1, 3, 1, 1))
        self.register_buffer('mean', torch.tensor(params['mean']).view(1, 3, 1, 1))

        self.loss_fn = loss_fn

    def forward(self, image):
        image = (image - self.mean) / self.std
        mask = self.model(image)

        return mask

    def training_step(self, batch):
        image, label = batch
        predict = self(image)

        loss = self.loss_fn(predict, label)
        self.log('train_loss', loss, prog_bar=True, logger=True)

        return {'loss': loss}

    def validation_step(self, batch):
        image, label = batch
        predict = self(image)

        loss = self.loss_fn(predict, label)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return {'loss', loss}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=cfg.lr)

        if cfg.scheduler == 'CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.min_lr)
        elif cfg.scheduler == 'ConsineAnnealingWarmRestarts':
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.T_0, eta_min=cfg.min_lr)
        elif cfg.scheduler == 'ReduceLROnPlateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7, threshold=0.0001,
                                                       min_lr=cfg.min_lr, )
        elif cfg.scheduler == 'ExponentialLR':
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
        elif cfg.scheduler is None:
            scheduler = None

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }