'''
Main module for migrant based population model.
'''
from utils import r2_keras, r2_population, limit_loss, distance_matrix, correlation_loss, weighted_mae, iso_to_continent
from scenarios import Scenario, constant_distribution, gravity_distribution, na_gravity_distribution
from models import get_model, WRAP

from keras.optimizers import Adam
from keras.losses import mae
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
    parser.add_argument('-c', '--continent', type=str, default='', help='The continent to run inference on.')
    return parser.parse_args()

def main():
    if os.path.exists('.\\logs'):
        shutil.rmtree('.\\logs')
    args = parse_args()
    wrap = WRAP
    train_years = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013'][5:]
    dev_years = ['2011', '2012', '2013', '2014', '2015', '2016']#, '2017']
    dmat = distance_matrix('C:/Users/Sean/Documents/MATH_498/data/distance_matrix.txt',
                           'C:/Users/Sean/Documents/MATH_498/data/country_codes.csv')
    modtype = 'state' + ('less' * (1 - args.state_model))
    scene = Scenario(modtype, train_years = train_years, dev_years = dev_years, distance_matrix=dmat, mlimit = True)
    model, inner = get_model('C:/Users/Sean/Documents/MATH_498/code/' + modtype + '_weights.h5', (20,), wrap = wrap, load = not args.train, state = args.state_model)
    # Work on the loss function
    compile_args = dict(metrics = [r2_keras, r2_population], loss= lambda x, y: weighted_mae(x, y), optimizer = Adam(lr = 0.001))
    fit_args = dict(epochs = 200, batch_size = 64, callbacks = [TensorBoard(log_dir='.\\logs', histogram_freq=5), EarlyStopping(patience=15, restore_best_weights=True)])
    scene.set_network(model, compile_args, fit_args, noise = 10000, wrap = wrap, train = args.train)
    state_model = inner if args.state_model else None
    if args.continent:
        continent_map = iso_to_continent()
        sub_states = [iso for iso in continent_map.index if continent_map[iso].lower() == args.continent.lower()]
    else:
        sub_states = None
    scene.run(year = '2000', timesteps=20, movement_function=na_gravity_distribution, view=args.view, state_model=state_model, sub_states = sub_states)
    model_json = scene.network.to_json()
    with open('C:/Users/Sean/Documents/MATH_498/code/' + modtype + '_model.json', "w") as json_file:
        json.dump(model_json, json_file)
    scene.network.save_weights('C:/Users/Sean/Documents/MATH_498/code/' + modtype + '_weights.h5')

if __name__ == '__main__':
    main()