'''
Scenario object implementation.

Allows for the running of a world scenario under a network.
'''
import numpy as np
from states import State
from utils import noisier, wrap_years
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Scenario:
    def __init__(self, name, train_years, dev_years, network=None,
                            year = None, compile_args = None, fit_args = None, **kwargs):
        self.name = name
        self.network = network
        State.network = self.network
        self.states = State.from_countries_of_the_world()
        self.train_years = train_years
        self. dev_years = dev_years
        self.data_shape = [State.to_X_array(self.states[train_years[0]])[0].shape, State.to_y_array(self.states[train_years[0]])[0].shape]
        if network and compile_args and fit_args:
            self.init_train(compile_args, fit_args)
        elif network and not (compile_args and fit_args):
            raise TypeError('compile_args or fit_args not defined for network.')

    def init_train(self, compile_args, fit_args, noise = None, wrap = 3):
        training_states = np.random.binomial(1, 0.75, size = (167,)).astype(bool)
        X = np.vstack([State.to_X_array(self.states[yr])[training_states] for yr in self.train_years])
        Xdev = np.vstack([State.to_X_array(self.states[yr])[~training_states] for yr in self.dev_years])
        y = np.vstack([State.to_y_array(self.states[yr])[training_states] for yr in self.train_years])
        ydev = np.vstack([State.to_y_array(self.states[yr])[~training_states] for yr in self.dev_years])
        xScaler, yScaler = MinMaxScaler(), MinMaxScaler()
        print(X.shape, y.shape)
        self.network.compile(**compile_args)
        X, y = wrap_years(X, y, training_states.sum(), wrap = wrap)
        Xdev, ydev = wrap_years(Xdev, ydev, (~training_states).sum(), wrap = wrap)
        X = xScaler.fit_transform(X)
        Xdev = xScaler.transform(Xdev)
        y = yScaler.fit_transform(y)
        ydev = yScaler.transform(ydev)
        self.xScaler = State.xScaler = xScaler
        self.yScaler = State.yScaler = yScaler
        if noise:
            X, y = noisier(X, y, degree = 0.05, samples = noise)
        self.network.fit(X, y, validation_data = (Xdev, ydev), **fit_args)

    def set_network(self, network, compile_args, fit_args, noise = None, wrap = 3):
        self.network = network
        State.network = self.network
        self.init_train(compile_args, fit_args, noise = noise, wrap = wrap)

    def write_out(self):
        print(*[pop.population for pop in self._running], sep = ',', file = self.poutfile)
        print(*[pop.migrants for pop in self._running], sep = ',', file = self.moutfile)
        print(*[pop.deaths for pop in self._running], sep = ',', file = self.doutfile)
        print(*[pop.births for pop in self._running], sep = ',', file = self.boutfile)

    def run(self, year = '2000', timesteps = 100):
        self._running = self.states[year]
        self.poutfile = open('generated_data/' + year + '_' + self.name + '_p_out.csv', 'w')
        self.moutfile = open('generated_data/' + year + '_' + self.name + '_m_out.csv', 'w')
        self.doutfile = open('generated_data/' + year + '_' + self.name + '_d_out.csv', 'w')
        self.boutfile = open('generated_data/' + year + '_' + self.name + '_b_out.csv', 'w')
        print(*[pop.name for pop in self._running], sep = ',', file = self.poutfile)
        print(*[pop.name for pop in self._running], sep = ',', file = self.moutfile)
        print(*[pop.name for pop in self._running], sep = ',', file = self.doutfile)
        print(*[pop.name for pop in self._running], sep = ',', file = self.boutfile)
        self.write_out()
        for t in tqdm(range(timesteps)):
            for pop in self._running:
                pop.timestep()
                pop.recalculate()
            migrations = [pop.migrants for pop in self._running]
            moving = -sum(m for m in migrations if m < 0)
            accepting = sum(m for m in migrations if m > 0)
            acceptors = [max(0, m/accepting) for m in migrations]
            for i, pop in enumerate(self._running):
                if pop.migrants > 0:
                    pop.migrants = acceptors[i] * moving
            self.write_out()