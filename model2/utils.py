'''
Functional utilities for scenarios and network training.
'''
import numpy as np
import keras.backend as K

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

def change_loss(y_true, y_pred):
    pass