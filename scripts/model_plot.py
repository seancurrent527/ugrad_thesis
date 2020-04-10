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
from keras.layers import Dense, Input, Concatenate, Dropout, Lambda
from sklearn.preprocessing import MinMaxScaler
from models import get_model
from gravity_regression import iso_to_continent

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--state_model', action='store_true')
    parser.add_argument('-c', '--continent', type = str, default = '')
    return parser.parse_args()

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
    print("ALL : Population R2 -", r2_population(targets.values, predictions))
    for index in feats.columns:
        cfeats = feats.copy()
        ctargets = targets.copy()
        cfeats[index].values[:] = np.random.rand(len(feats))
        ctargets[index].values[:] = np.random.rand(len(feats))
        predictions = model.predict(cfeats.values)
        print(f"{index} : Population R2 -", r2_population(ctargets.values, predictions))

def plot_2000(targs, preds, countries):
    testing_states = [c in countries for c in preds.columns]
    testing_targs = targs.iloc[:, testing_states]
    testing_preds = preds.iloc[1:, testing_states]
    for i in range(len(testing_targs.columns) // 16 + bool(len(testing_targs.columns) % 16)):
        fig = plt.figure(figsize=(8,8), frameon=False)
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

def get_world():
    world = gpd.read_file('C:/Users/Sean/Documents/MATH_498/data/map/ne_110m_admin_0_countries.shp')
    fix = {'Norway': 'NOR', 'France': 'FRA', 'Northern Cyprus': 'CYP', 'Somaliland': 'SOM', 'Kosovo': 'RKS'}
    for row in world.index:
        if world.loc[row, 'NAME_LONG'] in fix:
            world.loc[row, 'ISO_A3'] = fix[world.loc[row, 'NAME_LONG']]
    return world

def main():
    args = _parse_args()
    modtype = 'subset_state' + ('less' * (1 - args.state_model))
    feats = pd.read_pickle('C:/Users/Sean/Documents/MATH_498/code/country_data.pkl')
    year_data_p = pd.read_csv('C:/Users/Sean/Documents/MATH_498/code/generated_data/2000_' + modtype + '_p_out.csv')
    year_data_m = pd.read_csv('C:/Users/Sean/Documents/MATH_498/code/generated_data/2000_' + modtype + '_m_out.csv')
    year_data_d = pd.read_csv('C:/Users/Sean/Documents/MATH_498/code/generated_data/2000_' + modtype + '_d_out.csv')
    year_data_b = pd.read_csv('C:/Users/Sean/Documents/MATH_498/code/generated_data/2000_' + modtype + '_b_out.csv')
    
    countries = ['AUS', 'USA', 'RUS', 'CAN', 'AFG', 'BRA', 'DEU', 'FRA',
                 'GBR', 'CHN', 'IND', 'ARE', 'SAU', 'MEX', 'ESP', 'CHE']
    #countries = ['IND', 'ARG', 'COL', 'CHL', 'IDN', 'PNG', 'PHL', 'COD',
     #            'ANG', 'CAF', 'COG', 'NIG']
    continent_map = iso_to_continent()
    if args.continent:
        countries = [iso for iso in continent_map.index if continent_map[iso].lower() == args.continent.lower()]
    
    targets = feats
    p_targs = pd.DataFrame(data = np.array([df['SP.POP.TOTL'].values for _, df in targets.groupby(level = 0)]).T,
                           columns = year_data_p.columns)
    m_targs = pd.DataFrame(data = np.array([df['SM.POP.NETM'].values for _, df in targets.groupby(level = 0)]).T,
                           columns = year_data_m.columns)
    d_targs = pd.DataFrame(data = np.array([df['SP.DYN.CDRT.IN'].values for _, df in targets.groupby(level = 0)]).T,
                           columns = year_data_d.columns)
    b_targs = pd.DataFrame(data = np.array([df['SP.DYN.CBRT.IN'].values for _, df in targets.groupby(level = 0)]).T,
                           columns = year_data_b.columns)
    
    np.random.seed(147)
    plot_2000(p_targs, year_data_p, countries)
    np.random.seed(147)
    plot_2000(m_targs, year_data_m, countries)
    np.random.seed(147)
    plot_2000(d_targs, year_data_d, countries)
    np.random.seed(147)
    plot_2000(b_targs, year_data_b, countries)
    
    phl = targets['SP.DYN.CDRT.IN'].loc['PHL', :'2015'].values
    chl = targets['SP.DYN.CBRT.IN'].loc['CHL', :'2015'].values
    phl_preds = year_data_d['PHL'].iloc[:len(phl)].values
    chl_preds = year_data_b['CHL'].iloc[:len(chl)].values
    yrs = [i + 2000 for i in range(len(phl))]
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(yrs, phl, color = 'tab:blue')
    axes[0].plot(yrs, phl_preds, color = 'tab:orange', linestyle = '--')
    axes[0].set_title('Death Rate in the Philippines', fontsize = 16)
    axes[0].set_xlabel('Year', fontsize = 12)
    axes[0].set_ylabel('Death Rate (deaths per 1000 people)', fontsize = 12)
    axes[0].set_xticks([2000, 2005, 2010, 2015])
    axes[0].set_xlim([2000, 2015])
    axes[0].legend(['Observed', 'Predicted'])
    axes[1].plot(yrs, chl, color = 'tab:blue')
    axes[1].plot(yrs, chl_preds, color = 'tab:orange', linestyle = '--')
    axes[1].set_title('Birth Rate in Chile', fontsize = 16)
    axes[1].set_xlabel('Year', fontsize = 12)
    axes[1].set_ylabel('Birth Rate (births per 1000 people)', fontsize = 12)
    axes[1].legend(['Observed', 'Predicted'])
    axes[1].set_xticks([2000, 2005, 2010, 2015])
    axes[1].set_xlim([2000, 2015])
    plt.show()



if __name__ == '__main__':
    main()