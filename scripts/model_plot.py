import numpy as np
import matplotlib.pyplot as plt
import argparse
import importlib
import geopandas as gpd
from tqdm import tqdm
import pandas as pd
import keras.backend as K
from keras.models import Model
from keras.regularizers import l1
from keras.layers import Dense, Input, Concatenate, Dropout
from sklearn.preprocessing import MinMaxScaler

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type = str, default = 'complex')
    parser.add_argument('-f', '--function-model', type = str, default = 'model_network')
    return parser.parse_args()

def model_network(input_shape):
    xin = Input(input_shape)
    xhid1 = Dense(32, activation='relu', kernel_regularizer=l1(0.15))(xin)
    xhid1 = Dropout(0.5)(xhid1)
    xhid1 = Concatenate()([xin, xhid1])
    xhid2 = Dense(32, activation='relu', kernel_regularizer=l1(0.15))(xhid1)
    xhid2 = Dropout(0.5)(xhid2)
    xhid2 = Concatenate()([xhid1, xhid2])
    xhid3 = Dense(32, activation='relu', kernel_regularizer=l1(0.15))(xhid2)
    xhid3 = Dropout(0.5)(xhid3)
    xhid3 = Concatenate()([xhid2, xhid3])
    xout = Dense(input_shape[0])(xhid3)
    return Model(xin, xout)

def get_model(fname, input_shape):
    model = model_network(input_shape)
    model.load_weights(fname)
    return model

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def r2_score(y_true, y_pred):
    SS_res =  np.sum(np.square(y_true - y_pred))
    SS_tot = np.sum(np.square(y_true - np.mean(y_true))) 
    return (1 - SS_res/(SS_tot + 0.0000001))

def r2_population(y_true, y_pred):
    y_true, y_pred = y_true[:, :4], y_pred[:, :4]
    return r2_score(y_true, y_pred)

def test_effects(model, feats, targets):
    scaler = MinMaxScaler()
    feats = feats.copy()
    targets = targets.copy()
    feats.values[:] = scaler.fit_transform(feats.values)
    targets.values[:] = scaler.transform(targets)
    predictions = model.predict(feats.values)
    #print(targets)
    #print(predictions)
    print("ALL : Population R2 -", r2_population(targets.values, predictions))
    for index in feats.columns:
        cfeats = feats.copy()
        ctargets = targets.copy()
        cfeats[index].values[:] = np.random.rand(len(feats))
        ctargets[index].values[:] = np.random.rand(len(feats))
        predictions = model.predict(cfeats.values)
        #print(ctargets)
        #print(predictions)
        print(f"{index} : Population R2 -", r2_population(ctargets.values, predictions))

def plot_2000(targs, preds):
    training_states = np.random.binomial(1, 0.75, size = (167,)).astype(bool)
    training_targs = targs.iloc[:, training_states]
    training_preds = preds.iloc[1:, training_states]
    testing_targs = targs.iloc[:, ~training_states]
    testing_preds = preds.iloc[1:, ~training_states]
    for i in range(len(training_targs.columns) // 16 + 1):
        fig = plt.figure(figsize=(8,8))
        cut = i * 16
        t, p = training_targs.iloc[:, cut:cut + 16], training_preds.iloc[:, cut:cut + 16]  
        for j in range(16):
            try:
                sub = plt.subplot(4, 4, j + 1, label = t.columns[j])
            except IndexError:
                break
            sub.plot(t.iloc[:, j])
            sub.plot(p.iloc[:, j])
            sub.set_title(t.columns[j])
        plt.tight_layout()
        plt.show()
    for i in range(len(testing_targs.columns) // 16 + 1):
        fig = plt.figure(figsize=(8,8))
        cut = i * 16
        t, p = testing_targs.iloc[:, cut:cut + 16], testing_preds.iloc[:, cut:cut + 16]  
        for j in range(16):
            try:
                sub = plt.subplot(4, 4, j + 1, label = t.columns[j])
            except IndexError:
                break
            sub.plot(t.iloc[:, j])
            sub.plot(p.iloc[:, j])
            sub.set_title(t.columns[j])
        plt.tight_layout()
        plt.show()


def main():
    args = _parse_args()
    feats = pd.read_pickle('complex_features.pkl')
    targets = pd.read_pickle('complex_targets.pkl')
    model = get_model('complex_weights.h5', (len(feats.columns),))
    year_data_p = pd.read_csv('2000_' + args.type + '_p_out.csv')
    year_data_m = pd.read_csv('2000_' + args.type + '_m_out.csv')
    p_targs = pd.DataFrame(data = np.array([df['SP.POP.TOTL'].values for _, df in targets.groupby(level = 0)]).T,
                           columns = year_data_p.columns)
    m_targs = pd.DataFrame(data = np.array([df['SM.POP.NETM'].values for _, df in targets.groupby(level = 0)]).T,
                           columns = year_data_m.columns)
    #test_effects(model, feats, targets)
    np.random.seed(147)
    plot_2000(p_targs, year_data_p)
    np.random.seed(147)
    plot_2000(m_targs, year_data_m)

if __name__ == '__main__':
    main()