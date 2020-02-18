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
from scipy.stats.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
from mpl_toolkits.mplot3d import Axes3D
from model_plot import get_world

TABS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type = str, default = 'complex')
    return parser.parse_args()

def correlate_countries(predictiondf, targetdf, duration = 10):
    # Columns: Countries
    # Rows   : Years
    preds, targs = predictiondf.iloc[:duration], targetdf.iloc[:duration]
    corrs = pd.Series([spearmanr(preds[country].values, targs[country].values)[0] for country in targs.columns], targs.columns)
    #mae = pd.Series([abs(preds[country].values - targs[country].values).mean() for country in targs.columns], targs.columns)
    #mape = pd.Series([np.mean(np.abs((preds[country].values - targs[country].values) / targs[country].values)) for country in targs.columns], targs.columns)
    return corrs ** 2

def correlation_vectors(corr_results):
    df = pd.concat(corr_results, axis = 1)
    df.columns = ['p', 'm', 'd', 'b']
    return df

def cluster_me(corr_vectors, k = 4):
    corr_vectors = corr_vectors.values
    clusters = np.random.uniform(-1.0, 1.0, size = (k, corr_vectors.shape[1]))
    while True:
        #print(clusters)
        cluster_distances = np.column_stack([((corr_vectors - point)**2).sum(axis = 1) for point in clusters])
        #print(cluster_distances)
        closest_cluster = cluster_distances.argmin(axis = 1)
        #print(closest_cluster)
        new_clusters = np.array([corr_vectors[closest_cluster == i].mean(axis = 0) for i in range(k)])
        if (new_clusters == clusters).all():
            break
        clusters = new_clusters
    return new_clusters, closest_cluster

def plot_me(corr_vectors, cluster_labels):
    colors = [TABS[i] for i in cluster_labels]
    fig = plt.figure(figsize=(9, 6))
    cols_pairs = ['pm', 'pd', 'pb', 'db', 'mb', 'md']
    conversion = {'p': 'population', 'm': 'migration', 'd': 'death rate', 'b': 'birth rate'}
    for i, pr in enumerate(cols_pairs):
        sub = plt.subplot(2, 3, i + 1)
        sub.scatter(corr_vectors[pr[0]], corr_vectors[pr[1]], color = colors, marker = '.')
        sub.set_xlabel(conversion[pr[0]] + ' correlation')
        sub.set_ylabel(conversion[pr[1]] + ' correlation')
    plt.tight_layout()
    plt.show()

def est_density(arr, sample):
    kd = KernelDensity()
    kd.fit(arr.reshape(-1, 1))
    return kd.score_samples(sample.reshape(-1, 1))

def main():
    args = _parse_args()
    targets = pd.read_pickle('C:/Users/Sean/Documents/MATH_498/code/complex_targets.pkl')
    year_data_p = pd.read_csv('C:/Users/Sean/Documents/MATH_498/code/generated_data/2000_' + args.type + '_p_out.csv')
    year_data_m = pd.read_csv('C:/Users/Sean/Documents/MATH_498/code/generated_data/2000_' + args.type + '_m_out.csv')
    year_data_d = pd.read_csv('C:/Users/Sean/Documents/MATH_498/code/generated_data/2000_' + args.type + '_d_out.csv')
    year_data_b = pd.read_csv('C:/Users/Sean/Documents/MATH_498/code/generated_data/2000_' + args.type + '_b_out.csv')
    p_targs = pd.DataFrame(data = np.array([df['SP.POP.TOTL'].values for _, df in targets.groupby(level = 0)]).T,
                           columns = year_data_p.columns)
    m_targs = pd.DataFrame(data = np.array([df['SM.POP.NETM'].values for _, df in targets.groupby(level = 0)]).T,
                           columns = year_data_m.columns)
    d_targs = pd.DataFrame(data = np.array([df['SP.DYN.CDRT.IN'].values for _, df in targets.groupby(level = 0)]).T,
                           columns = year_data_d.columns)
    b_targs = pd.DataFrame(data = np.array([df['SP.DYN.CBRT.IN'].values for _, df in targets.groupby(level = 0)]).T,
                           columns = year_data_b.columns)
    p_corr = correlate_countries(year_data_p, p_targs)
    m_corr = correlate_countries(year_data_m, m_targs)
    d_corr = correlate_countries(year_data_d, d_targs)
    b_corr = correlate_countries(year_data_b, b_targs)
    corr_vecs = correlation_vectors([p_corr, m_corr, d_corr, b_corr])
    print(corr_vecs)
    corr_vecs = corr_vecs.fillna(0)

    world = get_world()
    world.index = world['ISO_A3']
    world[['p_corr', 'm_corr', 'd_corr', 'b_corr']] = corr_vecs[['p', 'm', 'd', 'b']]
    fig = plt.figure(figsize = (10,6))
    ax = plt.subplot(221)
    world.plot('p_corr', cmap = 'RdBu', ax=ax)
    ax.set_title('Population $R^{2}$')
    ax = plt.subplot(222)
    world.plot('m_corr', cmap = 'RdBu', ax=ax)
    ax.set_title('Migration $R^{2}$')
    ax = plt.subplot(223)
    world.plot('d_corr', cmap = 'RdBu', ax=ax)
    ax.set_title('Death Rate $R^{2}$')
    ax = plt.subplot(224)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    world.plot('b_corr', legend=True, cmap = 'RdBu', ax=ax, cax = cbar_ax)
    ax.set_title('Birth Rate $R^{2}$')
    plt.show()
    '''
    clust = OPTICS()
    clust.fit(corr_vecs.values)
    reachability = clust.reachability_[clust.ordering_]
    plt.plot(reachability, color = 'black', linestyle = '', marker = '.')
    plt.show()
    #clusters, labels = cluster_me(corr_vecs, k = 4) # k = 3 is consistently good, k = 4 is sometimes more revealing but noisier
    #for l in sorted(set(labels)):
     #   print(TABS[l][4:], ':', corr_vecs.index[labels == l])
    #plot_me(corr_vecs, labels)
   
    for _ in range(0):
        tsne = TSNE(1)
        transformedt = tsne.fit_transform(corr_vecs.values)
        pca = PCA(1)
        transformed = pca.fit_transform(corr_vecs.values)
        samp = np.linspace(transformed.min() - 5, transformed.max() + 5, 1000)
        sampt = np.linspace(transformed.min() - 5, transformed.max() + 5, 1000)
        
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], marker = '.', color = 'black')
        
        plt.hist(transformed, color = 'black', bins = 40)
        plt.show()
        plt.hist(transformedt, color = 'black', bins = 40)
        plt.show()
        
        #plt.plot(transformed, marker = '.', color = 'black', linestyle = '')
        #plt.title('PCA')
        #plt.show()
        plt.plot(transformedt, marker = '.', color = 'black', linestyle = '')
        plt.title('tsne')
        plt.show()
        
        #plt.plot(samp, np.exp(est_density(transformed, samp)))
        #plt.title('PCA')
        #plt.show()
        plt.plot(sampt, np.exp(est_density(transformedt, sampt)))
        plt.title('tsne')
        plt.show()
    '''

if __name__ == '__main__':
    main()