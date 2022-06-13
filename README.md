#### Image segmentation competitions

##### competition introduction url

https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/overview

##### preprocess train data

```bash
python preprocess_data.py
``` 

the bash file will add train image path to original train.csv

##### todolist

- 不对input使用transformers，对比实验效果
- pytorch-lighting trainer precision=16设置是为什么
- 增加n_fold处理

##### experiments record

| data preprocess                                              | fold | arch | encoder  | pretrained | transformers | loss function              | valid loss | submit score |
|--------------------------------------------------------------|------|------|----------|------------|--------------|----------------------------|------------|--------------|
| original grey image combine channels img img_size=(360, 384) | 单模   |unet             | resnet34 | imagenet   | yes          | 0.5BSELoss + 0.5TverskyLoss | 0.11649    | 0.834        |
| 2.5D img_size=(160, 194)                                     | 3折   |unet             | resnet34 | imagenet   | yes          | 0.5BSELoss + 0.5TverskyLoss | 0.13349    | 0.762        |

