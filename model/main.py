'''
Main module for migrant based population model.
'''
from utils import r2_keras, r2_population, ConCurrent, limit_loss, distance_matrix
from scenarios import Scenario, constant_distribution, gravity_distribution
from models import get_model

from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
import keras.backend as K

import numpy as np, os
import json, shutil
import argparse

np.random.seed(147)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true', help='Whether or not to retrain the neural network.')
    parser.add_argument('-v', '--view', action='store_true', help='Whether or not to view the migration information.')
    parser.add_argument('-s', '--state-model', action='store_true', help='Whether or not to use a state-based model for inference.')
    return parser.parse_args()

def weighted_mae(y_true, y_pred, weight = 1, wrap = 3):
    weights = np.ones((20 * wrap,))
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
    if os.path.exists('.\\logs'):
        shutil.rmtree('.\\logs')
    args = parse_args()
    wrap = 4
    train_years = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013']
    dev_years = ['2011', '2012', '2013', '2014', '2015', '2016']#, '2017']
    dmat = distance_matrix('C:/Users/Sean/Documents/MATH_498/data/distance_matrix.txt',
                           'C:/Users/Sean/Documents/MATH_498/data/country_codes.csv')
    modtype = 'state' + ('less' * (1 - args.state_model))
    scene = Scenario(modtype, train_years = train_years, dev_years = dev_years, distance_matrix=dmat, mlimit = True)
    model, inner = get_model('C:/Users/Sean/Documents/MATH_498/code/' + modtype + '_weights.h5', (20,), wrap = wrap, load = not args.train, state = args.state_model)
    # Work on the loss function
    compile_args = dict(metrics = [r2_keras, r2_population], loss='mse', optimizer = Adam(lr = 0.0001))
    fit_args = dict(epochs = 100, batch_size = 64, callbacks = [TensorBoard(log_dir='.\\logs', histogram_freq=5), EarlyStopping(patience=15, restore_best_weights=True)])
    scene.set_network(model, compile_args, fit_args, noise = 10000, wrap = wrap, train = args.train)
    state_model = inner if args.state_model else None
    scene.run(year = '2000', timesteps=20, movement_function=gravity_distribution, view=args.view, state_model=state_model)
    model_json = scene.network.to_json()
    with open('C:/Users/Sean/Documents/MATH_498/code/' + modtype + '_model.json', "w") as json_file:
        json.dump(model_json, json_file)
    scene.network.save_weights('C:/Users/Sean/Documents/MATH_498/code/' + modtype + '_weights.h5')

if __name__ == '__main__':
    main()