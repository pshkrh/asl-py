# Imports
from fastai import *
from fastai.vision import *
import fastai

tfms = get_transforms()
bs = 64
path = '/enter/path/to/dataset/here/'

data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=224, bs=bs).normalize(imagenet_stats)

learn = create_cnn(data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(6)

learn.lr_find()

learn.recorder.plot()

learn.unfreeze()

learn.fit_one_cycle(8, max_lr=slice(1e-5,1e-4))

learn.export()