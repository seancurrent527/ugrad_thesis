'''
State object implementation.

This is an agent basis for a population migration model.
'''
import os, json, requests
import pandas as pd, numpy as np
from collections import defaultdict

class State:

    network = None

    def __init__(self, name, year, targets = None, features = None):
        self.name = name
        self.year = year
        self.targets = targets
        self.features = features
        self.population = self.features['SP.POP.TOTL']
        self.migrants = self.features['SM.POP.NETM']
        self.deaths = self.features['SP.DYN.CDRT.IN']
        self.births = self.features['SP.DYN.CBRT.IN']

    def timestep(self):
        self.population = self.population + self.population * (self.births - self.deaths) / 1000 + self.migrants

    def recalculate(self):
        #self.features['SP.POP.TOTL'] = self.population
        #self.features['EN.POP.DNST'] = self.population / self.features['AG.LND.TOTL.K2']
        if State.network:
            adjusted = State.network.predict(State.xScaler.transform(State.to_X_array([self])))
            adjusted = State.yScaler.inverse_transform(adjusted)
            self.migrants, self.deaths, self.births = adjusted[0][0:3]
            self.features.values[:] = adjusted[0][:len(self.features)]

    @classmethod
    def from_countries_of_the_world(cls):
        features = pd.read_pickle('complex_features_20.pkl')
        targets = pd.read_pickle('complex_targets_20.pkl')
        states = defaultdict(list)
        for name, year in features.index:
            st = cls(name, year, targets = targets.loc[(name, str(int(year) + 1)), :], features = features.loc[(name, year), :])
            states[year].append(st)
        return states

    @staticmethod
    def to_X_array(states):
        return np.array([st.features.values for st in states])

    @staticmethod
    def to_y_array(states):
        return np.array([st.targets.values for st in states])