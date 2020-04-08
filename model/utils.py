'''
Functional utilities for scenarios and network training.
'''
import numpy as np, pandas as pd, random
import keras.backend as K
from keras.layers import Layer
from keras import initializers, regularizers, constraints, activations, losses
import geopandas as gpd
from models import WRAP
from sklearn.preprocessing import StandardScaler

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

def limit_loss(y_true, y_pred):
    return K.sqrt(losses.mean_squared_error(y_true, y_pred)) + losses.mean_absolute_error(y_true, y_pred)

def noisier(X, y, degree = 0.01, samples = 1000):
    Xdata = [*X]
    ydata = [*y]
    for i in range(samples - len(y)):
        j = random.randrange(0, len(y))
        noise_X = np.random.uniform(-degree, degree, len(Xdata[0])) + 1
        noise_y = np.random.uniform(-degree, degree, len(ydata[0])) + 1
        Xdata.append(X[j] * noise_X)
        ydata.append(y[j] * noise_y)
    return np.array(Xdata), np.array(ydata)

def r2_keras(y_true, y_pred):
    y_true, y_pred = K.reshape(y_true, (-1, WRAP, 20)), K.reshape(y_pred, (-1, WRAP, 20))
    SS_res =  K.sum(K.square(y_true - y_pred), axis = 1)
    SS_tot = K.sum(K.square(y_true - K.repeat(K.mean(y_true, axis = 1), WRAP)), axis = 1)
    return K.mean((1 - SS_res/(SS_tot + K.epsilon())))

def r2_population(y_true, y_pred):
    y_true, y_pred = K.reshape(y_true, (-1, WRAP, 20)), K.reshape(y_pred, (-1, WRAP, 20))
    y_true, y_pred = y_true[...,:4], y_pred[..., :4]
    SS_res =  K.sum(K.square(y_true - y_pred), axis = 1)
    SS_tot = K.sum(K.square(y_true - K.repeat(K.mean(y_true, axis = 1), WRAP)), axis = 1)
    return K.mean((1 - SS_res/(SS_tot + K.epsilon())))


def weighted_mae(y_true, y_pred):
    weights = np.ones((20,))
    all_weights = []
    for i in range(WRAP):
        all_weights.append((1 - (0.1 * i)) * weights)
    weights = np.concatenate(all_weights)
    weights = K.constant(weights)
    mae = K.abs(y_true - y_pred)
    return K.sum(mae * weights)

def correlation_loss(y_true, y_pred):
    # want to maximize correlation
    y_true, y_pred = K.reshape(y_true, (-1, WRAP, 20)), K.reshape(y_pred, (-1, WRAP, 20))
    mx = K.repeat(K.mean(y_true, axis = 1), WRAP)
    my = K.repeat(K.mean(y_pred, axis = 1), WRAP)
    xm, ym = y_true-mx, y_pred-my
    r_num = K.sum(xm * ym, axis = 1)
    r_den = K.sum(K.sum(K.square(xm), axis = 1) * K.sum(K.square(ym), axis = 1))
    r = r_num / r_den
    return 1 - r

def wrap_years(X, y, num_countries, wrap):
    year_arrays = [y[i * num_countries: (i + 1) * num_countries] for i in range(len(y) // num_countries)]
    concat_new_X = []
    concat_new_y = []
    for i in range(len(year_arrays) - wrap):
        concat_new_y.append(np.concatenate(year_arrays[i: i + wrap], axis = -1))
        concat_new_X.append(X[i * num_countries: (i + 1) * num_countries])
    new_X = np.concatenate(concat_new_X, axis = 0)
    new_y = np.concatenate(concat_new_y, axis = 0)
    return new_X, new_y

def get_world():
    world = gpd.read_file('C:/Users/Sean/Documents/MATH_498/data/map/ne_110m_admin_0_countries.shp')
    fix = {'Norway': 'NOR', 'France': 'FRA', 'Northern Cyprus': 'CYP', 'Somaliland': 'SOM', 'Kosovo': 'RKS'}
    for row in world.index:
        if world.loc[row, 'NAME_LONG'] in fix:
            world.loc[row, 'ISO_A3'] = fix[world.loc[row, 'NAME_LONG']]
    return world

def iso_to_continent():
    world = get_world().drop_duplicates('ISO_A3')
    world.index = world['ISO_A3']
    return world['CONTINENT']

def distance_matrix(distance_file, iso_file):
    distance_matrix = pd.read_csv(distance_file, index_col=0, keep_default_na=False, na_values=[])
    iso_matrix = pd.read_csv(iso_file, skiprows=[0], header = None, names = ['index', 'country', 'iso2', 'iso3'])
    two_to_three = pd.Series(iso_matrix['iso3'].values, iso_matrix['iso2'])
    two_to_three['SD'] = 'SDN'
    two_to_three['SS'] = 'SSD'
    two_to_three['NA'] = 'NAM'
    two_to_three['XK'] = 'RKS'
    three_codes = []
    errors = ['GZ']#, 'XK']
    distance_matrix = distance_matrix.drop(labels = errors, axis = 1)
    distance_matrix = distance_matrix.drop(labels = errors, axis = 0)
    for c in distance_matrix.index:
        try:
            three_codes.append(two_to_three[c])
        except:
            print(c)
    distance_matrix.index = three_codes
    distance_matrix.columns = three_codes
    return distance_matrix

def social_distance_matrix(features):
    features = pd.DataFrame(StandardScaler().fit_transform(features),
                            columns = features.columns, index = features.index)
    sdmat = pd.DataFrame(0, columns = features.index, index = features.index)
    euclidean_distance = lambda x, y: ( (x - y) ** 2 ).sum() ** 0.5
    for col in sdmat.columns:
        for row in sdmat.index:
            sdmat.loc[row, col] = euclidean_distance(features.loc[row], features.loc[col])
    return sdmat