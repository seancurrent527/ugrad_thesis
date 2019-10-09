'''
Scenario object implementation.

Allows for the running of a world scenario under a network.
'''
import numpy as np
from states import State
from utils import noisier
from tqdm import tqdm

class Scenario:
    def __init__(self, name, train_years, dev_years, network=None, outfile=None, **kwargs):
        self.name = name
        self.network = network
        State.network = self.network
        self.outfile = outfile if outfile else open(self.name + '_out.csv', 'w')
        self.states = State.from_countries_of_the_world()
        self.data_shape = [State.to_X_array([self.states[0]])[0].shape, State.to_y_array([self.states[0]])[0].shape]
        if network:
            self.init_train(**kwargs)

    def init_train(self, train_years, dev_years, **kwargs):
        #FIX THIS!!!
        X = np.vstack([State.to_X_array(self.states[yr]) for yr in train_years])
        Xdev = np.vstack([State.to_X_array(self.states[yr]) for yr in dev_years])
        y = np.vstack([State.to_y_array(self.states[yr]) for yr in train_years])
        ydev = np.vstack([State.to_y_array(self.states[yr]) for yr in dev_years])
        print(X.shape, y.shape)
        self.network.fit(X, y, validation_data = (Xdev, ydev), **kwargs)

    def set_network(self, network, **kwargs):
        self.network = network
        State.network = self.network
        self.init_train(**kwargs)

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
            moving = -sum(m for m in migrations if m < 0)
            accepting = sum(m for m in migrations if m > 0)
            acceptors = [max(0, m/accepting) for m in migrations]
            for i, pop in enumerate(self._running):
                if pop.migrants > 0:
                    pop.migrants = acceptors[i] * moving