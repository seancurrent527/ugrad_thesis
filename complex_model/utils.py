'''
Functional utilities for scenarios and network training.
'''
import numpy as np
import keras.backend as K
from keras.layers import Layer
from keras import initializers, regularizers, constraints, activations

class ConCurrent(Layer):
    def __init__(self, units, repeats,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ConCurrent, self).__init__(**kwargs)
        assert type(units) in (int, list), "units must be a list or integer"
        if type(units) is int:
            assert repeats is not None, "if units is an integer, repeats must be specified"
            self.units = [units for _ in range(repeats)]
        else:
            self.units = units
        self.repeats = len(self.units)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.kernels = []
        self.biases = []
        in_units = input_shape[-1]
        for i, out_units in enumerate(self.units):
            self.kernels.append(self.add_weight(name = 'kernel_' + str(i),
                                    shape = (in_units, out_units),
                                    initializer = self.kernel_initializer,
                                    regularizer = self.kernel_regularizer,
                                    constraint = self.kernel_constraint,
                                    trainable = True))
            if self.use_bias:
                self.biases.append(self.add_weight(name = 'bias_' + str(i),
                                    shape = (out_units,),
                                    initializer = self.bias_initializer,
                                    regularizer = self.bias_regularizer,
                                    constraint = self.bias_constraint,
                                    trainable = True))
            in_units = in_units + out_units
        super(ConCurrent, self).build(input_shape)

    def call(self, inputs):
        for i, kernel in enumerate(self.kernels):
            output = K.dot(inputs, kernel)
            if self.use_bias:
                output = K.bias_add(output, self.biases[i], data_format='channels_last')
            if self.activation is not None:
                output = self.activation(output)
            inputs = K.concatenate([inputs, output])
        return inputs

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = sum(self.units) + input_shape[-1]
        return tuple(output_shape)

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