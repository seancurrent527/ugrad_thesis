'''
Scenario object implementation.

Allows for the running of a world scenario under a network.
'''
import numpy as np
from states import State
from utils import noisier
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

class Scenario:
    def __init__(self, name, train_years, dev_years, network=None, outfile=None,
                            year = None, compile_args = None, fit_args = None, **kwargs):
        self.name = name
        self.network = network
        State.network = self.network
        self.outfile = outfile if outfile else open(self.name + '_out.csv', 'w')
        self.states = State.from_countries_of_the_world()
        self.train_years = train_years
        self. dev_years = dev_years
        self.data_shape = [State.to_X_array(self.states[train_years[0]])[0].shape, State.to_y_array(self.states[train_years[0]])[0].shape]
        if network and compile_args and fit_args:
            self.init_train(compile_args, fit_args)
        elif network and not (compile_args and fit_args):
            raise TypeError('compile_args or fit_args not defined for network.')

    def init_train(self, compile_args, fit_args):
        X = np.vstack([State.to_X_array(self.states[yr]) for yr in self.train_years])
        Xdev = np.vstack([State.to_X_array(self.states[yr]) for yr in self.dev_years])
        y = np.vstack([State.to_y_array(self.states[yr]) for yr in self.train_years])
        ydev = np.vstack([State.to_y_array(self.states[yr]) for yr in self.dev_years])
        xScaler, yScaler = StandardScaler(), StandardScaler()
        X = xScaler.fit_transform(X)
        Xdev = xScaler.transform(Xdev)
        y = yScaler.fit_transform(y)
        ydev = yScaler.transform(ydev)
        self.xScaler = State.xScaler = xScaler
        self.yScaler = State.yScaler = yScaler
        print(X.shape, y.shape)
        self.network.compile(**compile_args)
        self.network.fit(X, y, validation_data = (Xdev, ydev), **fit_args)

    def set_network(self, network, compile_args, fit_args):
        self.network = network
        State.network = self.network
        self.init_train(compile_args, fit_args)

    def write_out(self):
        print(*[pop.population for pop in self._running], sep = ',', file = self.outfile)

    def run(self, timesteps = 100, year = '2000'):
        self._running = self.states[year]
        print(*[pop.name for pop in self._running], sep = ',', file = self.outfile)
        for t in tqdm(range(timesteps)):
            for pop in self._running:
                pop.timestep()
                pop.recalculate()
            self.write_out()
            migrations = [pop.migrants for pop in self._running]
            #print(migrations)
            moving = -sum(m for m in migrations if m < 0)
            accepting = sum(m for m in migrations if m > 0)
            acceptors = [max(0, m/accepting) for m in migrations]
            for i, pop in enumerate(self._running):
                if pop.migrants > 0:
                    pop.migrants = acceptors[i] * moving