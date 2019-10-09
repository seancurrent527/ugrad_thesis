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
    xout = Dense(4)(xhid)
    return Model(xin, xout)

def test1():
    scene = Scenario('test1')
    inshape, outshape = scene.data_shape
    model = test1_network(inshape)
    scene.set_network(model, epochs = 30, metrics = [r2_keras], loss='mse')
    scene.run()

if __name__ == '__main__':
    test1()