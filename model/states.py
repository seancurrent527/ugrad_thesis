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
        self.population = self.features['SP.POP.TOTL'] // 1000 # Scale to blocks of 1000
        self.migrants = self.targets['SM.POP.NETM'] // 1000 # Scale to blocks of 1000
        self.births = self.targets['SP.DYN.CDRT.IN']
        self.deaths = self.targets['SP.DYN.CBRT.IN']

    def timestep(self):
        self.population = self.population + self.births - self.deaths + self.migrants

    def recalculate(self):
        self.features['SP.POP.TOTL'] = self.population * 1000
        self.features['EN.POP.DNST'] = self.population * 1000 / self.features['AG.LND.TOTL.K2']
        if State.network:
            adjusted = State.network.predict(State.xScaler.transform(State.to_X_array([self])))
            adjusted = State.yScaler.inverse_transform(adjusted)
            self.migrants, self.births, self.deaths = adjusted[0]

    @classmethod
    def from_countries_of_the_world(cls):
        features = pd.read_pickle('features.pkl')
        targets = pd.read_pickle('targets.pkl')
        states = defaultdict(list)
        for name, year in targets.index:
            st = cls(name, year, targets = targets.loc[(name, year), :], features = features.loc[(name, year), :])
            states[year].append(st)
        return states

    @staticmethod
    def to_X_array(states):
        return np.array([st.features.values for st in states])

    @staticmethod
    def to_y_array(states):
        return np.array([st.targets.values for st in states])
    
    ### DEPRECATED ###
    '''
    @staticmethod
    def to_json(fp, states):
        dct = {st:st.stats for st in states}
        json.dump(dct, fp)

    @classmethod
    def from_json(cls, fp):
        dct = json.load(fp)
        states = []
        for stname in dct:
            st = cls(stname)
            st.stats = dct[stname]
            states.append(st)
        return states
    
    @staticmethod
    def region_mapping():
        df = csv_to_dataframe('C:/Users/Sean/Documents/Thesis/migration/data/misc/countries_of_the_world.csv')
        format_df(df)
        regions = set(df['Region'])
        return {r:i for i,r in enumerate(sorted(regions))}

    @staticmethod
    def categorize_states(states):
        mp = State.region_mapping()
        for st in states:
            st.stats['Region'] = mp[st.stats['Region']]
    '''
#===========================================================================================

### DEPRECATED ###

'''
def csv_from_web(csvname, url, tnum):
    #https://en.wikipedia.org/wiki/Democracy_Index, 'wikitable sortable'
    #https://en.wikipedia.org/wiki/World_Happiness_Report, 'wikitable sortable'
    if os.path.exists(csvname):
        os.remove(csvname)
    r = requests.get(url)
    df = pd.read_html(r.text)[tnum]
    df.drop_duplicates().to_csv(csvname, header = 0, index = False)

def csv_to_dataframe(csvname):
    return pd.read_csv(csvname, decimal = ',', index_col = 0)
    
def format_df(df):
    rows, cols = df.shape
    for i in range(rows):
        for j in range(cols):
            if type(df.iloc[i, j]) == str:
                df.iloc[i, j] = df.iloc[i, j].strip().title()
    df.index = [c.strip().lower().replace(',', '') for c in df.index]
    df.dropna(inplace = True)

def init_setup(fname):
    states = State.from_countries_of_the_world()
    State.categorize_states(states)
    with open(fname, 'w') as fp:
        State.to_json(fp, states)
'''