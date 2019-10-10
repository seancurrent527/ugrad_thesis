'''
Main module for migrant based population model.
'''
from keras.models import Model
from utils import r2_keras
from scenarios import Scenario
from keras.layers import Dense, Input, Lambda
from keras.optimizers import Adam
import os

def test1_network(input_shape):
    xin, xpopin = Input(input_shape), Input((1,))
    xhid = Dense(128, activation = 'relu')(xin)
    xhid = Dense(64, activation = 'relu')(xhid)
    xout = Dense(3, activation = 'tanh')(xhid)
    
    xpopout = Lambda(lambda x: x[0] + x[1] + x[0] * (x[2] - x[3]))([xpopin, xout[0], xout[2], xout[1]])
    
    return Model([xin, xpopin], [xout, xpopout])

def test1():
    train_years = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009']
    dev_years = ['2010', '2011', '2012']
    test_years = ['2013', '2014', '2015']
    scene = Scenario('2013', train_years = train_years, dev_years = dev_years)
    inshape, outshape = scene.data_shape
    model = test1_network(inshape)
    compile_args = dict(metrics = [r2_keras], loss='mse', optimizer = Adam(lr = 0.00001))
    fit_args = dict(epochs = 100)
    scene.set_network(model, compile_args, fit_args)
    scene.run()

if __name__ == '__main__':
    test1()