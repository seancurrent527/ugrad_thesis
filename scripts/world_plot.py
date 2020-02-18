import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import argparse

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type = str, default = 'complex')
    return parser.parse_args()

def plot_pop_growth(year_data):
    for i, population_frame in enumerate(year_data):
        population_frame.iloc[len(year_data) - i: - (i + 1)].sum(axis = 1).plot()
    plt.show()

def subplot_times(year_data):
    y14, y15, y16 = year_data
    for i in range(11):
        fig = plt.figure(figsize=(8,8))
        cut = i * 16
        d14 = y14.iloc[2:, cut: cut + 16]
        d15 = y15.iloc[1:-1, cut: cut + 16]
        d16 = y16.iloc[:-2, cut: cut + 16]
        for j in range(16):
            try:
                sub = plt.subplot(4, 4, j + 1, label = d14.columns[j])
            except IndexError:
                break
            sub.plot(d14.iloc[:, j])
            sub.plot(d15.iloc[:, j])
            sub.plot(d16.iloc[:, j])
            sub.set_title(d14.columns[j])
        plt.tight_layout()
        plt.show()

def main():
    args = _parse_args()
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    years = ['2014', '2015', '2016']
    year_data_p = [pd.read_csv('C:/Users/Sean/Documents/MATH_498/code/generated_data/' + yr + '_' + args.type + '_p_out.csv') for yr in years]
    year_data_m = [pd.read_csv('C:/Users/Sean/Documents/MATH_498/code/generated_data/' + yr + '_' + args.type + '_m_out.csv') for yr in years]
    duration = len(year_data_p[0])
    for i in range(len(years)):
        yr = int(years[i])
        new_index = range(yr, yr + duration)
        year_data_p[i].index = new_index
        year_data_m[i].index = new_index
    plot_pop_growth(year_data_p)
    subplot_times(year_data_p)
    subplot_times(year_data_m)

if __name__ == '__main__':
    main()