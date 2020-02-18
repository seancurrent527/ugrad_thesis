'''
State object implementation.

This is an agent basis for a population migration model.
'''
import os, json, requests
import pandas as pd, numpy as np
from collections import defaultdict

class State:

    network = None

    def __init__(self, name, year, targets = None, features = None, mlimit = None):
        self.name = name
        self.year = year
        self.targets = targets
        self.features = features
        self.mlimit = mlimit
        self.population = self.features['SP.POP.TOTL']
        self.migrants = self.features['SM.POP.NETM']
        self.deaths = self.features['SP.DYN.CDRT.IN']
        self.births = self.features['SP.DYN.CBRT.IN']

    def timestep(self):
        self.population = self.population + self.population * (self.migrants + self.births - self.deaths) / 1000
        self.features['SP.POP.TOTL'] = self.population

    def recalculate(self):
        if State.network:
            adjusted = State.network.predict(State.scaler.transform(State.to_X_array([self])))
            adjusted = State.scaler.inverse_transform(adjusted[:,:20])
            #Apply migration limit
            if self.mlimit is not None:
                adjusted[0, 0] = np.sign(adjusted[0, 0]) * min(self.mlimit, abs(adjusted[0, 0]))
            self.migrants, self.deaths, self.births = adjusted[0, 0:3]
            self.features.values[:] = adjusted[0]

    def adjust_migrants(self, value, method = 'set'):
        if method.lower() == 'set':
            self.migrants = self.features['SM.POP.NETM'] = value
        elif method.lower() == 'add':
            self.migrants = self.features['SM.POP.NETM'] = self.migrants + value
        elif method.lower() == 'scale':
            self.migrants = self.features['SM.POP.NETM'] = self.migrants * value
        else:
            raise ValueError('\'' + method + '\' is not a valid value for parameter \'method\'')

    @classmethod
    def from_data(cls, mlimit = True):
        features = pd.read_pickle('C:/Users/Sean/Documents/MATH_498/code/complex_features.pkl')
        targets = pd.read_pickle('C:/Users/Sean/Documents/MATH_498/code/complex_targets.pkl')
        states = defaultdict(list)
        migration_limits = features.abs().groupby(level = 0).max()['SM.POP.NETM']
        for name, year in features.index:
            st = cls(name, year, targets = targets.loc[(name, str(int(year) + 1)), :],
                                features = features.loc[(name, year), :])
            if mlimit:
                st.mlimit = migration_limits[name]
            states[year].append(st)
        return states

    @staticmethod
    def to_X_array(states):
        return np.array([st.features.values for st in states])

    @staticmethod
    def to_y_array(states):
        return np.array([st.targets.values for st in states])