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

cfg.use_25d = True
cfg.n_25d_shift = 1

cfg.seed = 314
cfg.DATASET_DIR = r'/home/zhuyuedlut/Datasets/uw-madison-gi-tract-image-segmentation'

cfg.n_fold = 5

cfg.img_size = [320, 320]

cfg.train_bs = 64
cfg.valid_bs = 32
cfg.workers = 24

cfg.num_classes = 3

cfg.lr = 3e-5
cfg.scheduler = 'CosineAnnealingLR'
cfg.min_lr = 1e-6
cfg.epochs = 25
cfg.T_max = 150
cfg.T_0 = 30
cfg.patience = 6

cfg.output_path = './checkpoints'

cfg.data_transforms = {
    "train": A.Compose([
        A.OneOf([
            # A.Resize(*[224, 224], interpolation=cv2.INTER_NEAREST, p=1.0),
            # A.Resize(*[256, 256], interpolation=cv2.INTER_NEAREST, p=1.0),
            # A.Resize(*[288, 288], interpolation=cv2.INTER_NEAREST, p=1.0),
            # A.Resize(*[320, 320], interpolation=cv2.INTER_NEAREST, p=1.0),
            # A.Resize(*[352, 352], interpolation=cv2.INTER_NEAREST, p=1.0),
            # A.Resize(*[384, 384], interpolation=cv2.INTER_NEAREST, p=1.0),
            A.Resize(*cfg.img_size, interpolation=cv2.INTER_NEAREST, p=1.0),
        ], p=1),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=90, p=0.5),
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
