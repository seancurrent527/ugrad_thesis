'''
Main module for migrant based population model.
'''
from keras.models import Model
from utils import r2_keras
from scenarios import Scenario
from keras.layers import Dense, Input, Concatenate
from keras.optimizers import Adam
from keras.regularizers import l2
import os

def test1_network(input_shape):
    xin = Input(input_shape)
    xhid = Dense(128, activation = 'relu', kernel_regularizer=l2(0.01))(xin)
    xhid = Dense(64, activation = 'relu', kernel_regularizer=l2(0.01))(xhid)
    xoutp = Dense(3)(xhid)
    xoutm = Dense(1)(xhid)
    xout = Concatenate()([xoutm, xoutp])
    return Model(xin, xout)

def test1():
    train_years = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009']
    dev_years = ['2010', '2011', '2012']
    test_years = ['2013', '2014', '2015', '2016', '2017']
    scene = Scenario('2013_simple', train_years = train_years, dev_years = dev_years)
    inshape, outshape = scene.data_shape
    model = test1_network(inshape)
    compile_args = dict(metrics = [r2_keras], loss='mse', optimizer = Adam(lr = 0.00001))
    fit_args = dict(epochs = 250)
    scene.set_network(model, compile_args, fit_args)
    scene.run(year = '2013')

if __name__ == '__main__':
    test1()