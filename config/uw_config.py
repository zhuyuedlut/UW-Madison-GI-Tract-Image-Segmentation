# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : https://github.com/zhuyuedlut
# @date       : 2022/5/15 21:44
# @brief      : uw config file
"""
import albumentations as A
import cv2

from easydict import EasyDict

cfg = EasyDict()

cfg.seed = 1234
cfg.DATASET_DIR = r'/home/zhuyuedlut/Datasets/uw-madison-gi-tract-image-segmentation-2.5d'

cfg.n_fold = 5

cfg.img_size = [352, 352]

cfg.train_bs = 32
cfg.valid_bs = 64
cfg.workers = 4

cfg.num_classes = 3

cfg.lr = 2e-5
cfg.scheduler = 'CosineAnnealingLR'
cfg.min_lr = 1e-6
cfg.epochs = 25
cfg.T_max = int(30000 / cfg.train_bs * cfg.epochs) + 50
cfg.T_0 = 30
cfg.patience = 6

cfg.output_path = './checkpoints'

cfg.data_transforms = {
    "train": A.Compose([
        A.OneOf([
            A.Resize(*cfg.img_size, interpolation=cv2.INTER_NEAREST, p=1.0),
        ], p=1),

        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
        ], p=0.25),
        A.CoarseDropout(max_holes=8, max_height=cfg.img_size[0] // 20, max_width=cfg.img_size[1] // 20,
                        min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
    ], p=1.0),

    "valid_test": A.Compose([
        A.Resize(*cfg.img_size, interpolation=cv2.INTER_NEAREST),
    ], p=1.0)
}
