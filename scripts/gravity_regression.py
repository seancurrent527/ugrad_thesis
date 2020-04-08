import pandas as pd, geopandas as gpd
import numpy as np
from data_cleaning import get_world
from scipy.optimize import curve_fit
import statsmodels.api as sm
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

STOCK_DATA = 'C:/Users/Sean/Documents/MATH_498/data/world_bank/intercountry_migration_1960-2000.csv'
COUNTRY_DATA = 'C:/Users/Sean/Documents/MATH_498/code/country_data.pkl'
DISTANCE_DATA = 'C:/Users/Sean/Documents/MATH_498/data/distance_matrix.txt'
ISO_DATA = 'C:/Users/Sean/Documents/MATH_498/data/country_codes.csv'

np.random.seed(147)

def get_data():
    names = ['origin_name', 'origin_code', 'gender_name', 'gender_code', 'dest_name', 'dest_code', '1960', '1970', '1980', '1990', '2000']
    stock = pd.read_csv(STOCK_DATA, names = names, header = None, na_values = ['..'])
    stock = stock[stock['gender_code'] == 'TOT'] #only work with total migrations
    mid = pd.MultiIndex.from_frame(stock[['origin_code', 'dest_code']])
    stock.index = mid
    migrations = stock_to_migrations(stock)
    countries = pd.read_pickle(COUNTRY_DATA)
    countries = countries.groupby(level = 1).get_group('2000')
    countries.index = [c for c, y in countries.index]
    return migrations, countries

def stock_to_migrations(stock):
    stock = stock.fillna(0)
    migrations = (stock['2000'].astype(int) - stock['1990'].astype(int)) / 10
    return migrations

def distance_matrix(distance_file, iso_file):
    distance_matrix = pd.read_csv(distance_file, index_col=0, keep_default_na=False, na_values=[])
    iso_matrix = pd.read_csv(iso_file, skiprows=[0], header = None, names = ['index', 'country', 'iso2', 'iso3'])
    two_to_three = pd.Series(iso_matrix['iso3'].values, iso_matrix['iso2'])
    two_to_three['SD'] = 'SDN'
    two_to_three['SS'] = 'SSD'
    two_to_three['NA'] = 'NAM'
    two_to_three['XK'] = 'RKS'
    three_codes = []
    errors = ['GZ']
    distance_matrix = distance_matrix.drop(labels = errors, axis = 1)
    distance_matrix = distance_matrix.drop(labels = errors, axis = 0)
    for c in distance_matrix.index:
        try:
            three_codes.append(two_to_three[c])
        except:
            print(c)
    distance_matrix.index = three_codes
    distance_matrix.columns = three_codes
    #distance_matrix /= distance_matrix.max().max()
    return distance_matrix

def social_distance_matrix(features):
    features = pd.DataFrame(StandardScaler().fit_transform(features),
                            columns = features.columns, index = features.index)
    sdmat = pd.DataFrame(0, columns = features.index, index = features.index)
    euclidean_distance = lambda x, y: ( (x - y) ** 2 ).sum() ** 0.5
    for col in tqdm(sdmat.columns):
        for row in sdmat.index:
            sdmat.loc[row, col] = euclidean_distance(features.loc[row], features.loc[col])
    return sdmat

def gravity_distribution(x, sdp, gdp, dr, c, b, p1p, p2p):
    p1, p2, gd, sd, sgn = x.T
    denom = dr * sd**sdp + (1 - dr) * gd**gdp
    #denom = sd**sdp * gd**gdp
    num = p1**p1p * p2**p2p
    return sgn * c * num / denom + b

def iso_to_continent():
    world = get_world().drop_duplicates('ISO_A3')
    world.index = world['ISO_A3']
    return world['CONTINENT']

def r_squared(y_true, y_pred):
    return 1 - sum((y_true - y_pred) ** 2)/sum((y_true - y_true.mean()) ** 2)

def rmse(y_true, y_pred):
    return ((y_true - y_pred)**2).sum()**0.5

def cpc(y_true, y_pred):
    common = np.array([y_true, y_pred]).min(axis = 0)
    return common.sum() / (y_true.sum() + y_pred.sum())

def data_arrays(features, migrations, dmat, sdmat):
    x, y = [], []
    for s_row in tqdm(sdmat.index):
        for d_col in sdmat.columns:
            if s_row != d_col:
                feats = []
                feats.append(features['SP.POP.TOTL'][s_row])
                feats.append(features['SP.POP.TOTL'][d_col])
                feats.append(dmat[s_row][d_col])
                feats.append(sdmat[s_row][d_col])
                feats.append(np.sign(migrations[s_row, d_col]))
                x.append(feats)
                y.append(migrations[s_row, d_col])
    return np.array(x), np.array(y)

def fit_distribution(x, y):
    p = 5
    bounds = ([-p, -p, 0, 0, -np.inf, -p, -p],
              [p, p, 1, np.inf, np.inf, p, p])
    try:
        popt, _ = curve_fit(gravity_distribution, x, y, bounds = bounds,
        p0 = [1, 1, 0.5, 1, 0, 1, 1])
    except RuntimeError:
        print('Could not complete.')
        return False

    def dist_function(x):
        return gravity_distribution(x, *popt)

    pred = dist_function(x)
    return dist_function, popt, r_squared(y, pred), cpc(y, pred), rmse(y, pred)

def from_to_function(features, dmat, sdmat, dist_function):
    
    def from_to(s_row, d_col, sgn):
        if s_row == d_col:
            return 0
        feats = []
        feats.append(features['SP.POP.TOTL'][s_row])
        feats.append(features['SP.POP.TOTL'][d_col])
        feats.append(dmat[s_row][d_col])
        feats.append(sdmat[s_row][d_col])
        feats.append(sgn)
        feats = np.array(feats)
        return dist_function(feats)

    return from_to

def plot_pvm(x, y):
    p_product = []
    migs = []
    for i in range(len(x)):
        p_product.append(x[i][0] * x[i][1])
        migs.append(y[i])
    plt.plot(p_product, migs, linestyle = '', marker = '.')

def main():
    migrations, features = get_data()
    gdmat = distance_matrix(DISTANCE_DATA, ISO_DATA)
    sdmat = social_distance_matrix(features)
    conts = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America', 'ALL']
    continents = iso_to_continent()
    for c in conts:
        print(c)
        if c == 'ALL':
            countries = features.index
            gdmat_ = gdmat
            sdmat_ = sdmat
        else:
            countries = list(filter(lambda x: x in features.index, continents.index[continents == c]))
            gdmat_ = gdmat.loc[countries, countries]
            sdmat_ = sdmat.loc[countries, countries]
        x, y = data_arrays(features, migrations, gdmat_, sdmat_)
        result = fit_distribution(x, y)
        if not result:
            continue
        dist_func, params, r2, cpc_, loss = result
        print(params)
        print(r2)
        print(cpc_)
        print(loss)
        from_to = from_to_function(features, gdmat_, sdmat_, dist_func)
        with open('C:/Users/Sean/Documents/MATH_498/code/generated_data/' + '_'.join(c.lower().split()) + '_migrations.csv', 'w') as fp:
            print('From', 'To', 'Migrations', sep = '\t', file=fp)
            for s in countries:
                for d in countries:
                    print(s, d, from_to(s, d, np.sign(migrations[s, d])), sep = '\t', file=fp)
    #plot_pvm(x, y)
    #plt.show()
    #plt.plot(x[:, 2], y, linestyle = '', marker = '.')
    #plt.show()
    #plt.plot(x[:, 3], y, linestyle = '', marker = '.')
    #plt.show()

if __name__ == '__main__':
    main()