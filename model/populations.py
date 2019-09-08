'''
Populus object implementation.

Populus objects represent the population of a state.
'''
from states import State

class Population:

    network = None

    def __init__(self, state):
        self.state = state
        self.name = self.state.name
        self.population = self.state.stats['Population']//1000 # to scale with birth/death rates
        self.migrants = self.state.stats['Net migration']
        self.births = self.state.stats['Birthrate']
        self.deaths = self.state.stats['Deathrate']
        self.survival = self.state.stats['Infant mortality']

    def timestep(self):
        self.population = self.population + self.births - self.survival - self.deaths + self.migrants

    def recalculate(self):
        self.state.stats['Population'] = self.population * 1000
        self.state.stats['Pop. Density'] = self.population * 1000 / self.state.stats['Area']
        if Population.network:
            adjusted = Populus.network.predict(State.to_X_array([self.state]))
            self.migrants, self.births, self.deaths, self.survival = adjusted[0]