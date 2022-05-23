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

| arch | encoder  | pretrained| transformers | loss function              | valid loss | submit score |
|------|----------|------------|--------------|----------------------------|------------|--------------|
| unet | resnet34 | no         | no           | 0.5BSELoss + 0.5TverskyLoss | todo       | todo         |

