#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import keras
from generator import DataGenerator
from models import testModel
from keras import Model
import pickle
#from keras.utils import plot_model

def dist():
    return 0


## Data prep
df = pd.read_csv('../ct-img-classification/train_data_190726.csv', usecols=['Fpath','Labels'])
df = df[~((df.Labels=='unkw') | (df.Labels=='leg'))] # shape: (8785, 2)
gen = DataGenerator(df.Fpath.values, batch_size=16, shuffle=False)

## Build model
net = testModel()
f_extractor = Model(net.model.input, net.model.layers[-2].output)
f_extractor.load_weights('190805_190726_multi_TestModel_20.h5', by_name=True)

## Calculate feature map
f_maps =  f_extractor.predict_generator(gen, use_multiprocessing=True, verbose=1)
data = {'Fpath':df.Fpath.values, 'Features':f_maps}
pickle.dump(data, open('190726_DATA_fmaps.'), 'wb')
