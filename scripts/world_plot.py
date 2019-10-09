import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv('C:/Users/Sean/Documents/Thesis/migration/data/misc/countries_of_the_world.csv')
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    migration_series = pd.read_csv('C:/Users/Sean/Documents/Thesis/migration/code/model/test1_out.csv')
    print(data)
    print(world)
    print(migration_series)
    world.plot(figsize = (8, 4))
    plt.show()

if __name__ == '__main__':
    main()