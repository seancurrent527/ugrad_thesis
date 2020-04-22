'''
Scenario object implementation.

Allows for the running of a world scenario under a network.
'''
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
from states import State
from utils import noisier, wrap_years, get_world, social_distance_matrix
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

def constant_distribution(states, distance_matrix):
    countries = [pop.name for pop in states]
    movements = pd.DataFrame(0, index = countries, columns = countries)
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
    countries = [pop.name for pop in states]
    movements = pd.DataFrame(0, index = countries, columns = countries)
    features = pd.DataFrame({pop.name: pop.features for pop in states})
    geo_distance = distance_matrix / distance_matrix.max().max()
    sq_distance_function = lambda x, y: ((features[x] - features[y])**2).sum() + geo_distance.loc[x, y]**2
    gravity_function = lambda x, y: features[x]['SP.POP.TOTL'] * features[y]['SP.POP.TOTL'] / sq_distance_function(x, y)
    for source in movements.index:
        for destination in movements.columns:
            if source != destination:
                movements.loc[source, destination] = mc * (features[destination]['SM.POP.NETM'] - features[source]['SM.POP.NETM']) / 1000 * gravity_function(source, destination)
    return movements

def na_gravity_distribution(states, distance_matrix, mc = 1):
    countries = [pop.name for pop in states]
    movements = pd.DataFrame(0, index = countries, columns = countries)
    features = pd.DataFrame({pop.name: pop.features for pop in states})
    social_distance = social_distance_matrix(features.transpose())
    geo_distance = distance_matrix / distance_matrix.max().max()
    
    def gravity_function(source, destination):
        p1, p2 = features[source]['SP.POP.TOTL'], features[destination]['SP.POP.TOTL']
        gd = geo_distance.loc[source, destination]
        sd = social_distance.loc[source, destination]
        #sdp, gdp, dr, c, b, p1p, p2p = [2.91969034e+00, 3.09641037e+00, 1.24497981e-01, 5.90492723e-19,
         #                               -1.84534024e+03, 9.18318206e-01, 3.12928645e+00]
        sdp, gdp, dr, c, b, p1p, p2p = [5.00000000e+00, 1.01878632e+00, 8.52035094e-04, 6.67198465e-02,
        7.53251485e+00, 4.21096346e-01, 6.04829088e-01]
        denom = dr * sd**sdp + (1 - dr) * gd**gdp
        num = p1**p1p * p2**p2p
        return c * num / denom + b

    for source in countries:
        for destination in countries:
            if source != destination:
                movements.loc[source, destination] = gravity_function(source, destination)
    return movements

def view_migrations(dist_function, country = 'USA'):
    
    def distribution_function(*args, **kwargs):
        movements = dist_function(*args, **kwargs)
        world = get_world()
        world.index = world['ISO_A3']
        world = world.loc[movements.index]
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
        scaler = StandardScaler()
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

    def run(self, year = '2000', timesteps = 100, reset_period = None, movement_function = constant_distribution, mc = 1, view = True, state_model = None, sub_states = None):
        state = False
        if state_model is not None:
            state = True
            State.state_network = state_model
        self._running = self.states[year]
        if sub_states:
            self._running = [state for state in self._running if state.name in sub_states]
        separator = ('_subset' * bool(sub_states)) + '_' + ('with_resets_' * bool(reset_period))
        self.poutfile = open('C:/Users/Sean/Documents/MATH_498/code/generated_data/' + year + separator + self.name + '_p_out.csv', 'w')
        self.moutfile = open('C:/Users/Sean/Documents/MATH_498/code/generated_data/' + year + separator + self.name + '_m_out.csv', 'w')
        self.doutfile = open('C:/Users/Sean/Documents/MATH_498/code/generated_data/' + year + separator + self.name + '_d_out.csv', 'w')
        self.boutfile = open('C:/Users/Sean/Documents/MATH_498/code/generated_data/' + year + separator + self.name + '_b_out.csv', 'w')
        self.sdoutfile = open('C:/Users/Sean/Documents/MATH_498/code/generated_data/' + year + separator + self.name + '_sd_out.csv', 'w')
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
                pop.recalculate(state = state)
            features = pd.DataFrame({pop.name: pop.features for pop in self._running})
            sd = social_distance_matrix(features.transpose())
            values = np.array([sd.values[i, j] for i in range(len(sd)) for j in range(len(sd)) if i < j])
            print(*values, sep = ',', file = self.sdoutfile)
            if view:
                movements = view_migrations(movement_function)(self._running, self.distance_matrix)
            else:
                movements = movement_function(self._running, self.distance_matrix)
            for pop in self._running:
                #NOTE: inclusion of model causes WILD oscillations or convergence to a constant.
                #migrants = movements.loc[:, pop.name].sum() - movements.loc[pop.name].sum()
                #migrants = (migrants / pop.population)# * 1000
                #print(migrants)
                #pop.adjust_migrants(migrants, method='set')
                pop.timestep()
            self.write_out()