import numpy as np
import matplotlib.pyplot as plt
import argparse
import importlib
import geopandas as gpd
from tqdm import tqdm
import pandas as pd
import keras.backend as K
from keras.models import Model, load_model, model_from_json
from keras.regularizers import l1
from keras.layers import Dense, Input, Concatenate, Dropout, Lambda
from sklearn.preprocessing import MinMaxScaler

def sub_network(input_shape):
    xin = Input(input_shape)
    xhid1 = Dense(32, activation='relu', kernel_regularizer=l1(0.15))(xin)
    xhid1 = Dropout(0.5)(xhid1)
    xhid2 = Concatenate()([xin, xhid1])
    xout = Dense(input_shape[0])(xhid2)

    xhid3 = Concatenate()([xout, xhid2])
    corr = Dense(input_shape[0], activation = 'tanh')(xhid3)
    
    xout = Lambda(lambda x: x[0] * x[1])([xout, corr])
    return Model(xin, xout)

def model_network(input_shape, wrap = 3):
    xin = Input(input_shape)
    subnet = sub_network(input_shape)
    xhid = xin
    outs = []

    for _ in range(wrap):
        xhid = subnet(xhid)
        outs.append(xhid)
    
    if len(outs) > 1:
        xout = Concatenate()(outs)
    else:
        xout = outs[0]
    return Model(xin, xout)

def get_model(fname, input_shape):
    model = model_network(input_shape)
    model.load_weights(fname)
    return model

def plot_weights(weights):
    mats = [w for w in weights if len(w.shape) == 2]
    fig, ax = plt.subplots(1, len(mats), figsize = (10, 8))
    for i in range(len(mats)):
        im = ax[i].matshow(mats[i], cmap = 'RdBu', vmin = -0.8, vmax = 0.8)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    plt.colorbar(im, ax = ax, cax = cbar_ax)

def main():
    #with open('complex_model.json') as fp:
     #   model = model_from_json(fp.read())
    #model.load_weights('complex_weights.h5')
    model = get_model('C:/Users/Sean/Documents/MATH_498/code/complex_weights.h5', (20,))
    weights = model.get_weights()
    print([w.shape for w in weights])
    plot_weights(weights)
    plt.show()

if __name__ == '__main__':
    main()