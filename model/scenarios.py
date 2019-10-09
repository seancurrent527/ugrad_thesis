'''
Scenario object implementation.

Allows for the running of a world scenario under a network.
'''
from states import State
from utils import noisier
from tqdm import tqdm

class Scenario:
    def __init__(self, name, network=None, outfile=None, **kwargs):
        self.name = name
        self.network = network
        State.network = self.network
        self.outfile = outfile if outfile else open(self.name + '_out.csv', 'w')
        self.states = State.from_countries_of_the_world()
        State.categorize_states(self.states)
        print(*[pop.name for pop in self.states], sep = ',', file = self.outfile)
        self.data_shape = [State.to_X_array([self.states[0]])[0].shape, State.to_y_array([self.states[0]])[0].shape]
        if network:
            self.init_train(**kwargs)
        self.write_out()

    def init_train(self, **kwargs):
        X, y = noisier(State.to_X_array(self.states), State.to_y_array(self.states), degree = 0.05, samples = 2000)
        print(X.shape, y.shape)
        self.network.fit(X, y, **kwargs)

    def set_network(self, network, **kwargs):
        self.network = network
        State.network = self.network
        self.init_train(**kwargs)

    def write_out(self):
        print(*[pop.population for pop in self.states], sep = ',', file = self.outfile)

    def run(self, timesteps = 100):
        for t in tqdm(range(timesteps)):
            for pop in self.states:
                pop.timestep()
                pop.recalculate()
            self.write_out()
            migrations = [pop.migrants for pop in self.states]
            moving = -sum(m for m in migrations if m < 0)
            accepting = sum(m for m in migrations if m > 0)
            acceptors = [max(0, m/accepting) for m in migrations]
            for i, pop in enumerate(self.states):
                if pop.migrants > 0:
                    pop.migrants = acceptors[i] * moving