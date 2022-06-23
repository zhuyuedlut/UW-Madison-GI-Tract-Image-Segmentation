# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2022/5/15 22:16
# @brief      : uw segmentation不同的model实现
"""
import numpy as np
import torch
import torch.optim as optim
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from torch.optim import lr_scheduler
from config.uw_config import cfg

JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
DiceLoss = smp.losses.DiceLoss(mode='multilabel')
BCELoss = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)


def dice_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


def iou_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou


def loss_fn(y_pred, y_true):
    return 0.5 * BCELoss(y_pred, y_true) + 0.5 * DiceLoss(y_pred, y_true)


class UWModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, encoder_weights, classes, in_channels):
        super(UWModel, self).__init__()
        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )

        self.loss_fn = loss_fn

    def forward(self, image):
        pred = self.model(image)

        return pred

    def training_step(self, batch, _):
        image, mask = batch
        predict = self(image)

        loss = self.loss_fn(predict, mask)
        self.log('train_loss', loss, prog_bar=True, logger=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        predict = self(image)
        predict = torch.nn.Sigmoid()(predict)

        val_dice = dice_coef(mask, predict).cpu().detach().numpy()
        val_jaccard = iou_coef(mask, predict).cpu().detach().numpy()
        return [val_dice, val_jaccard]

    def validation_epoch_end(self, outputs) -> None:
        val_scores = np.mean(outputs, axis=0)
        val_dice, val_jaccard = val_scores
        self.log('val_dice', val_dice, prog_bar=True, logger=True)

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
