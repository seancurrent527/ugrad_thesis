'''
Scenario object implementation.

Allows for the running of a world scenario under a network.
'''
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
from states import State
from utils import noisier, wrap_years, get_world
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

def constant_distribution(states, distance_matrix):
    movements = pd.DataFrame(0, index = distance_matrix.index, columns = distance_matrix.columns)
    migrants = pd.Series({pop.name: pop.population * pop.migrants / 1000 for pop in states})
    populations = pd.Series({pop.name: pop.population for pop in states})
    acceptors = (migrants > 0) * migrants
    proportions = acceptors / acceptors.sum()
    for source in movements.index:
        for destination in movements.columns:
            if source != destination:
                movements.loc[source, destination] = proportions[destination] * (-migrants[source]) / populations[destination] * 1000 #Because sources are negative
            else:
                movements.loc[source, destination] = 0
    return movements

def gravity_distribution(states, distance_matrix, mc = 1):
    movements = pd.DataFrame(0, index = distance_matrix.index, columns = distance_matrix.columns)
    features = pd.DataFrame({pop.name: pop.features for pop in states})
    geo_distance = distance_matrix / distance_matrix.max().max()
    sq_distance_function = lambda x, y: ((features[x] - features[y])**2).sum() + geo_distance.loc[x, y]**2
    gravity_function = lambda x, y: features[x]['SP.POP.TOTL'] * features[y]['SP.POP.TOTL'] / sq_distance_function(x, y)
    for source in movements.index:
        for destination in movements.columns:
            if source != destination:
                movements.loc[source, destination] = mc * (features[source]['SM.POP.NETM'] - features[destination]['SM.POP.NETM']) / 1000 * gravity_function(source, destination)
    return movements

def view_migrations(dist_function, country = 'USA'):
    
    def distribution_function(*args, **kwargs):
        movements = dist_function(*args, **kwargs)
        world = get_world()
        world.index = world['iso_a3']
        world['migrations'] = movements[country]
        world['migrations'] = world['migrations'].fillna(0.0)
        print(world['migrations'][country])
        print(list(world['migrations']))
        bound = max(abs(world['migrations'].max()), abs(world['migrations'].min()))
        values = [-bound, 0, bound]
        clrs = ['tab:red', 'white', 'tab:blue']
        norm = plt.Normalize(-bound, bound)
        tuples = list(zip(map(norm, values), clrs))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', tuples)
        world.plot('migrations', cmap=cmap, norm=norm)
        m=plt.cm.ScalarMappable(cmap=cmap)
        plt.colorbar(m, boundaries = np.linspace(-bound, bound, 101))
        plt.show()
        return movements

    def distribution_function_(*args, **kwargs):
        movements = dist_function(*args, **kwargs)
        shapename = 'admin_0_countries'
        countries_shp = shpreader.natural_earth(resolution='110m',
                                        category='cultural', name=shapename)
        ax = plt.axes(projection=ccrs.PlateCarree())
        migrations = movements['USA']
        bound = max(abs(migrations.max()), abs(migrations.min()))
        values = [-bound, 0, bound]
        clrs = ['tab:red', 'white', 'tab:blue']
        norm = plt.Normalize(-bound, bound)
        tuples = list(zip(map(norm, values), clrs))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', tuples)
        for country in shpreader.Reader(countries_shp).records():
            iso = country.attributes['ISO_A3']
            if iso not in migrations.index:
                migrations[iso] = 0.0
            ax.add_geometries((country.geometry,), ccrs.PlateCarree(), facecolor=cmap(norm(migrations[iso])), label=iso)
        m=plt.cm.ScalarMappable(cmap=cmap)
        plt.colorbar(m, boundaries = np.linspace(-bound, bound, 101))
        plt.show()
        return movements

    return distribution_function

class Scenario:
    def __init__(self, name, train_years, dev_years, network=None,
                            year = None, compile_args = None, fit_args = None,
                            distance_matrix = None, mlimit = True, **kwargs):
        self.name = name
        self.network = network
        State.network = self.network
        self.states = State.from_data(mlimit=mlimit)
        self.train_years = train_years
        self.dev_years = dev_years
        self.distance_matrix = distance_matrix
        state_names = {p.name for p in self.states[train_years[0]]}
        for i in range(2):
            remove = set(self.distance_matrix.index) - state_names
            self.distance_matrix = self.distance_matrix.drop(remove, axis = 1-i)
        self.data_shape = [State.to_X_array(self.states[train_years[0]])[0].shape, State.to_y_array(self.states[train_years[0]])[0].shape]
        if network and compile_args and fit_args:
            self.init_model(compile_args, fit_args)
        elif network and not (compile_args and fit_args):
            raise TypeError('compile_args or fit_args not defined for network.')

    def init_model(self, compile_args, fit_args, train = True, noise = None, wrap = 3):
        training_states = np.random.binomial(1, 0.6, size = (len(self.states[self.train_years[0]]),)).astype(bool)
        X = np.vstack([State.to_X_array(self.states[yr])[training_states] for yr in self.train_years])
        Xdev = np.vstack([State.to_X_array(self.states[yr])[~training_states] for yr in self.dev_years])
        y = np.vstack([State.to_y_array(self.states[yr])[training_states] for yr in self.train_years])
        ydev = np.vstack([State.to_y_array(self.states[yr])[~training_states] for yr in self.dev_years])
        scaler = MinMaxScaler()
        print(X.shape, y.shape)
        self.network.compile(**compile_args)
        X = scaler.fit_transform(X)
        Xdev = scaler.transform(Xdev)
        y = scaler.transform(y)
        ydev = scaler.transform(ydev)
        self.scaler = State.scaler = scaler
        X, y = wrap_years(X, y, training_states.sum(), wrap = wrap)
        Xdev, ydev = wrap_years(Xdev, ydev, (~training_states).sum(), wrap = wrap)
        if noise:
            X, y = noisier(X, y, degree = 0.05, samples = noise)
        if train:
            self.network.fit(X, y, validation_data = (Xdev, ydev), **fit_args)

    def set_network(self, network, compile_args, fit_args, train = True, noise = None, wrap = 3):
        self.network = network
        State.network = self.network
        self.init_model(compile_args, fit_args, train = train, noise = noise, wrap = wrap)

    def write_out(self):
        print(*[pop.population for pop in self._running], sep = ',', file = self.poutfile)
        print(*[pop.migrants for pop in self._running], sep = ',', file = self.moutfile)
        print(*[pop.deaths for pop in self._running], sep = ',', file = self.doutfile)
        print(*[pop.births for pop in self._running], sep = ',', file = self.boutfile)

    def run(self, year = '2000', timesteps = 100, reset_period = None, movement_function = constant_distribution, mc = 1, view = True):
        self._running = self.states[year]
        separator = '_' if reset_period is None else '_with_resets_'
        self.poutfile = open('C:/Users/Sean/Documents/MATH_498/code/generated_data/' + year + separator + self.name + '_p_out.csv', 'w')
        self.moutfile = open('C:/Users/Sean/Documents/MATH_498/code/generated_data/' + year + separator + self.name + '_m_out.csv', 'w')
        self.doutfile = open('C:/Users/Sean/Documents/MATH_498/code/generated_data/' + year + separator + self.name + '_d_out.csv', 'w')
        self.boutfile = open('C:/Users/Sean/Documents/MATH_498/code/generated_data/' + year + separator + self.name + '_b_out.csv', 'w')
        print(*[pop.name for pop in self._running], sep = ',', file = self.poutfile)
        print(*[pop.name for pop in self._running], sep = ',', file = self.moutfile)
        print(*[pop.name for pop in self._running], sep = ',', file = self.doutfile)
        print(*[pop.name for pop in self._running], sep = ',', file = self.boutfile)
        self.write_out()
        for t in tqdm(range(timesteps)):
            if reset_period and ((int(year) + t) % reset_period) == 0:
                self._running = self.states[str(int(year) + t + 1)]
                self.write_out()
                continue
            for pop in self._running:
                pop.recalculate()
            if view:
                movements = view_migrations(movement_function)(self._running, self.distance_matrix)
            else:
                movements = movement_function(self._running, self.distance_matrix)
            for destination in self._running:
                if destination.migrants > 0:
                    destination.adjust_migrants(0, method = 'set')
                    for source in self._running:
                        if source.name != destination.name:
                            destination.adjust_migrants(mc * movements[destination.name][source.name], method = 'add')
                if destination.migrants < 0:
                    destination.adjust_migrants(mc, method = 'scale')
            for pop in self._running:
                pop.timestep()
            self.write_out()