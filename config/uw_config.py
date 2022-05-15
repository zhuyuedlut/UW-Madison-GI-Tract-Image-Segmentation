# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2022/5/15 21:44
# @brief      : uw config file
"""

from easydict import EasyDict

cfg = EasyDict()

cfg.train_bs = 32
cfg.valid_bs = 64
cfg.workers = 0

cfg.num_classes = 3