'''
NetworkModel class implementation.

This class is a keras model wrapper built for population modeling.
'''
from keras.models import Model
from keras.layers import Input
import random
import numpy as np
import keras.backend as K

class NetworkModel:
    def __init__(self):
        self.input = None
        self.output = None

    def add(self, layer):
        if self.input is None:
            self.input = Input(shape = layer.batch_input_shape[1:])
        if self.output is None:
            self.output = layer(self.input)
        else:
            self.output = layer(self.output)

    def fit(self, X, y, optimizer = 'adam', loss = 'binary_crossentropy', metrics = [], **kwargs):
        self.model = Model(self.input, self.output)
        self.model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
        self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def evaluate(self, X, y, **kwargs):
        return self.model.evaluate(X, y, **kwargs)

def noisier(X, y, degree = 0.01, samples = 1000):
    Xdata = [*X]
    ydata = [*y]
    for i in range(samples - len(y)):
        j = random.randrange(0, len(y))
        noise_X = np.random.uniform(-degree, degree, len(Xdata[0])) + 1
        noise_y = random.uniform(-degree, degree) + 1
        Xdata.append(X[j] * noise_X)
        ydata.append(y[j] * noise_y)
    return np.array(Xdata), np.array(ydata)

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))