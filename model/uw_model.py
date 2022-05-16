# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2022/5/15 22:16
# @brief      : uw segmentation不同的model实现
"""
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp


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
        pass
