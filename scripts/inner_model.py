import numpy as np
import matplotlib.pyplot as plt
import argparse
import importlib
import geopandas as gpd
from tqdm import tqdm
import pandas as pd
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler
import models
from keras.utils import plot_model

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--state_model', action='store_true')
    return parser.parse_args()

def plot_weights(weights):
    mats = [w for w in weights if len(w.shape) == 2]
    fig, ax = plt.subplots(1, len(mats), figsize = (10, 8))
    for i in range(len(mats)):
        im = ax[i].matshow(mats[i], cmap = 'RdBu')#, vmin = -0.8, vmax = 0.8)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    plt.colorbar(im, ax = ax, cax = cbar_ax)

def main():
    args = _parse_args()
    modtype = 'state' + ('less' * (1 - args.state_model))
    model, inner = models.get_model('C:/Users/Sean/Documents/MATH_498/code/' + modtype + '_weights.h5', (20,), wrap = 4, state = args.state_model)
    weights = model.get_weights()
    #plot_model(model, to_file=modtype + 'model.png')
    #plot_model(inner, to_file=modtype + 'inner.png')
    print([w.shape for w in weights])
    plot_weights(weights)
    plt.show()

if __name__ == '__main__':
    main()