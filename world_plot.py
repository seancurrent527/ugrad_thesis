import geopandas as gpd
import matplotlib.pyplot as plt

def main():
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    print(world)
    world.plot(figsize = (8, 4))
    plt.show()

if __name__ == '__main__':
    main()