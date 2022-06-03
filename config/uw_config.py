# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2022/5/15 21:44
# @brief      : uw config file
"""

from easydict import EasyDict

cfg = EasyDict()

cfg.seed = 1234
cfg.DATASET_DIR = r'/mnt/datasets/uw-madison-gi-tract-image-segmentation'

cfg.n_fold = 5

cfg.width = 256
cfg.height = 256

cfg.train_bs = 32
cfg.valid_bs = 64
cfg.workers = 32

cfg.num_classes = 3

cfg.lr = 2e-5
cfg.scheduler = 'CosineAnnealingLR'
cfg.min_lr = 1e-6
cfg.epochs = 25
cfg.T_max = int(30000 / cfg.train_bs * cfg.epochs) + 50
cfg.T_0 = 30
cfg.patience = 6

cfg.output_path = './checkpoints'