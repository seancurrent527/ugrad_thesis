'''
Main module for migrant based population model.
'''
from keras.models import Model
from networks import r2_keras
from scenarios import Scenario
from keras.layers import Dense
import os

def test1_network(input_shape):
    xin = Input(input_shape)
    xhid = Dense(128, activation = 'relu')(xin)
    xhid = Dense(64, activation = 'relu')(xhid)
    xout = Dense(3)(xhid)
    return Model(xin, xout)

def test1():
    train_years = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009']
    dev_years = ['2010', '2011', '2012']
    test_years = ['2013', '2014', '2015']
    scene = Scenario('2013', train_years = train_years, dev_years = dev_years)
    inshape, outshape = scene.data_shape
    model = test1_network(inshape)
    scene.set_network(model, epochs = 30, metrics = [r2_keras], loss='mse')
    scene.run()

if __name__ == '__main__':
    test1()