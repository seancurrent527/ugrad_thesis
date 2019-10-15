import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import argparse

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type = str, default = 'simple')
    return parser.parse_args()

def main():
    args = _parse_args()
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    migration_series = pd.read_csv('C:/Users/Sean/Documents/Thesis/migration/code/2013_' + args.type + '_m_out.csv')
    population_series = pd.read_csv('C:/Users/Sean/Documents/Thesis/migration/code/2013_' + args.type + '_p_out.csv')
    #world.plot(figsize = (8, 4))
    #plt.show()
    population_series.sum(axis = 1).plot()
    plt.show()
    for col in migration_series:
        population_series[col].plot()
    plt.show()
    migration_series.sum(axis = 1).plot()
    plt.show()
    for col in migration_series:
        migration_series[col].plot()
    plt.show()
    

if __name__ == '__main__':
    main()