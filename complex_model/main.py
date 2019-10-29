'''
Main module for migrant based population model.
'''
from keras.models import Model
from utils import r2_keras, r2_population, ConCurrent, limit_loss
from scenarios import Scenario
from keras.layers import Dense, Input, Concatenate, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2, l1, l1_l2
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
import numpy as np, os
import keras.backend as K

np.random.seed(147)

def model_network(input_shape):
    xin = Input(input_shape)
    xhid1 = Dense(32, activation='relu', kernel_regularizer=l1(0.15))(xin)
    xhid1 = Dropout(0.5)(xhid1)
    xhid1 = Concatenate()([xin, xhid1])
    xhid2 = Dense(32, activation='relu', kernel_regularizer=l1(0.15))(xhid1)
    xhid2 = Dropout(0.5)(xhid2)
    xhid2 = Concatenate()([xhid1, xhid2])
    xhid3 = Dense(32, activation='relu', kernel_regularizer=l1(0.15))(xhid2)
    xhid3 = Dropout(0.5)(xhid3)
    xhid3 = Concatenate()([xhid2, xhid3])
    xout = Dense(input_shape[0])(xhid3)
    return Model(xin, xout)

def weighted_mae(y_true, y_pred):
    weights = np.ones((44,))
    weights[:4] *= 10
    wieghts = K.constant(weights)
    mae = K.abs(y_true - y_pred)
    return K.sum(mae * weights)

def test1():
    train_years = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010']
    dev_years = ['2011', '2012', '2013']
    test_years = ['2014', '2015', '2016']
    scene = Scenario('complex', train_years = train_years, dev_years = dev_years)
    inshape, outshape = scene.data_shape
    model = model_network(inshape)
    compile_args = dict(metrics = [r2_keras, r2_population], loss=weighted_mae, optimizer = Adam(lr = 0.0001))
    fit_args = dict(epochs = 300, batch_size = 64, callbacks = [TensorBoard(log_dir='.\logs', histogram_freq=5), EarlyStopping(patience=10, restore_best_weights=True)])
    scene.set_network(model, compile_args, fit_args, noise = 10000)
    scene.run(year = '2000', timesteps=50)
    for year in test_years:
        scene.run(year = year, timesteps=50)
    scene.network.save_weights('complex_weights.h5')

if __name__ == '__main__':
    test1()