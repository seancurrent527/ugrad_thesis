'''
Main module for migrant based population model.
'''
from network import NetworkModel, r2_keras
from world import Scenario
from keras.layers import Dense
import os

def test1_network(input_shape):
    model = NetworkModel()
    model.add(Dense(128, input_shape = input_shape, activation = 'tanh'))
    model.add(Dense(64, activation = 'tanh'))
    model.add(Dense(4))
    return model

def test1():
    scene = Scenario('test1')
    inshape, outshape = scene.data_shape
    model = test1_network(inshape)
    scene.set_network(model, epochs = 30, metrics = [r2_keras], loss='mse')
    scene.run()

if __name__ == '__main__':
    test1()