'''
State object implementation.

This is an agent basis for a population migration model.
'''
import os, json, requests
import pandas as pd, numpy as np

class State:

    targets = ['Net migration', 'Birthrate', 'Deathrate', 'Infant mortality']

    def __init__(self, name):
        self.name = name
        self.stats = {}

    @classmethod
    def from_countries_of_the_world(cls):
        df = csv_to_dataframe('../../data/countries_of_the_world.csv')
        format_df(df)
        states = []
        for name, row in df.iterrows():
            st = cls(name)
            st.stats = dict(row)
            states.append(st)
        return states

    @staticmethod
    def to_X_array(states):
        return np.array([[st.stats[key] for key in sorted(st.stats) if key not in State.targets] for st in states])

    @staticmethod
    def to_y_array(states):
        return np.array([[st.stats[t] for t in State.targets] for st in states])

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
        df = csv_to_dataframe('../../data/countries_of_the_world.csv')
        format_df(df)
        regions = set(df['Region'])
        return {r:i for i,r in enumerate(sorted(regions))}

    @staticmethod
    def categorize_states(states):
        mp = State.region_mapping()
        for st in states:
            st.stats['Region'] = mp[st.stats['Region']]

#===========================================================================================
def csv_from_web(csvname, url, tnum):
    #https://en.wikipedia.org/wiki/Democracy_Index, 'wikitable sortable'
    #https://en.wikipedia.org/wiki/World_Happiness_Report, 'wikitable sortable'
    if os.path.exists(csvname):
        os.remove(csvname)
    r = requests.get(url)
    df = pd.read_html(r.text)[tnum]
    df.drop_duplicates().to_csv(csvname, header = 0, index = False)

def csv_to_dataframe(csvname):
    '''
    reads in a csv to a dataframe
    '''
    return pd.read_csv(csvname, decimal = ',', index_col = 0)
    
def format_df(df):
    '''
    formats the dataframe in place
    '''
    rows, cols = df.shape
    for i in range(rows):
        for j in range(cols):
            if type(df.iloc[i, j]) == str:
                df.iloc[i, j] = df.iloc[i, j].strip().title()
    df.index = [c.strip() for c in df.index]
    df.dropna(inplace = True)

def init_setup(fname):
    states = State.from_countries_of_the_world()
    State.categorize_states(states)
    with open(fname, 'w') as fp:
        State.to_json(fp, states)

