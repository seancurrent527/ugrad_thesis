'''
Main module for migrant based population model.
'''
from keras.models import Model
from utils import r2_keras, r2_population, ConCurrent, limit_loss
from scenarios import Scenario
from keras.layers import Dense, Input, Concatenate, Dropout, Lambda
from keras.optimizers import Adam
from keras.regularizers import l2, l1, l1_l2
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
import numpy as np, os
import keras.backend as K

np.random.seed(147)

def sub_network(input_shape):
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

    xhid4 = Concatenate()([xout, xhid3])
    corr = Dense(input_shape[0], activation = 'tanh')(xhid4)
    scale = Dense(input_shape[0], activation = 'relu')(xhid4)
    
    xout = Lambda(lambda x: x[0] * x[1] * x[2])([xout, corr, scale])
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

def weighted_mae(y_true, y_pred, weight = 1):
    weights = np.ones((80,))
    weights[:4] *= weight
    weights = K.constant(weights)
    mae = K.abs(y_true - y_pred)
    return K.sum(mae * weights)

def correlation_loss(y_true, y_pred):
    # want to maximize correlation
    y_true, y_pred = K.reshape(y_true, (4, -1)), K.reshape(y_pred, (4, -1))
    mx = K.mean(y_true, axis = 0)
    my = K.mean(y_pred, axis = 0)
    xm, ym = y_true-mx, y_pred-my
    r_num = K.sum(xm * ym, axis = 0)
    r_den = K.sum(K.sum(K.square(xm), axis = 0) * K.sum(K.square(ym), axis = 0))
    r = r_num / r_den
    return 1 - r

def main():
    wrap = 4
    train_years = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013']
    dev_years = ['2011', '2012', '2013', '2014', '2015', '2016']
    test_years = ['2014', '2015', '2016']
    scene = Scenario('multiyear', train_years = train_years, dev_years = dev_years)
    inshape, _ = scene.data_shape
    model = model_network(inshape, wrap = wrap)
    # Work on the loss function
    compile_args = dict(metrics = [r2_keras, r2_population, correlation_loss], loss=lambda x, y: weighted_mae(x, y), optimizer = Adam(lr = 0.0001))
    fit_args = dict(epochs = 300, batch_size = 64, callbacks = [TensorBoard(log_dir='.\\logs', histogram_freq=5), EarlyStopping(patience=15, restore_best_weights=True)])
    scene.set_network(model, compile_args, fit_args, noise = 10000, wrap = wrap)
    scene.run(year = '2000', timesteps=20)
    for year in test_years:
        scene.run(year = year, timesteps=20)
    scene.network.save_weights('multiyear_weights.h5')

if __name__ == '__main__':
    main()