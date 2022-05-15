# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2022/5/15 22:16
# @brief      : uw segmentation不同的model实现
"""

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

        self.loss_fn = loss_fn

    def forward(self):
        pass



