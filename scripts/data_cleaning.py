'''
Data cleaning script for WDIData.
'''
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

DISCOVER_FEATS = False

TARGET_FEATS = ['SM.POP.NETM', 'SP.DYN.CDRT.IN', 'SP.DYN.CBRT.IN', 'SP.POP.TOTL']

NECESSARY_FEATS = ['SM.POP.NETM',           #0
                   'SP.DYN.CDRT.IN',        #1
                   'SP.DYN.CBRT.IN',        #2
                   'SP.POP.TOTL',           #3
                   'SP.POP.DPND',           #4
                   'VC.BTL.DETH',           #5
                   'SL.TLF.TOTL.IN',        #6
                   'MS.MIL.XPND.GD.ZS',     #7
                   'EN.POP.DNST',           #8
                   'SP.DYN.LE00.MA.IN',     #9
                   'SP.RUR.TOTL.ZS',        #10
                   'SP.URB.TOTL.IN.ZS',     #11
                   'SP.URB.TOTL',           #12
                   'SP.RUR.TOTL',           #13
                   'EN.ATM.METH.AG.KT.CE',  #14
                   'SP.DYN.AMRT.FE',        #15
                   'SP.RUR.TOTL.ZG',        #16
                   'SP.DYN.TFRT.IN',        #17
                   'EN.ATM.NOXE.AG.KT.CE',  #18
                   'SP.DYN.LE00.IN']        #19

FEATURES = ['EG.ELC.ACCS.ZS',       # - access to electricity (also has rural/urban)
            'SE.PRM.TENR',          # - percent enrolled primary education
            'NY.ADJ.NNTY.KD.ZG',    # - adjusted net national income
            'SP.ADO.TFRT',          # - adolescent fertility rate
            'SP.POP.DPND',          # - age dependency ratio
            'AG.LND.AGRI.K2',       # - agricultural land area
            #'EN.ATM.METH.AG.ZS',    # - ag. methane emissions
            #'EN.ATM.NOXE.AG.ZS',    # - ag. no2 emissions
            'EG.USE.COMM.CL.ZS',    # - alt. and nuclear energy usage
            'AG.LND.ARBL.ZS',       # - percent arable land
            'MS.MIL.TOTL.TF.ZS',    # - percent military personnel
            'VC.BTL.DETH',          # - battle related deaths
            'SL.TLF.0714.ZS',       # - percent children in employment
            'FP.CPI.TOTL',          # - consumer price index
            'EG.USE.ELEC.KH.PC',    # - electric power consumption
            'SL.EMP.TOTL.SP.NE.ZS', # - employment ratio
            #'SP.DYN.TFRT.IN',       # - fertility rate
            'NY.GDP.PCAP.KD',       # - gdp per capita
            'SI.POV.GINI',          # - gini index estimate
            'IT.NET.USER.ZS',       # - percent using the internet
            'SL.TLF.TOTL.IN',       # - total labor force
            'AG.LND.TOTL.K2',       # - total land area
            #'SP.DYN.LE00.IN',       # - life expectancy at birth
            #'EN.ATM.METH.KT.CE',    # - methane emissions
            'MS.MIL.XPND.GD.ZS',    # - percent military expenditure
            #'EN.ATM.NOXE.KT.CE',    # - no2 emissions
            'SH.MED.PHYS.ZS',       # - physicians per 1000
            'EN.POP.DNST']          # - population density
            #'SP.POP.TOTL',          # - total population
            #'EG.ELC.RNEW.ZS',       # - percent renewable energy
            #'SP.RUR.TOTL',          # - Rural Population
            #'SL.UEM.TOTL.NE.ZS',    # - total unemployment
            #'SP.URB.TOTL'}          # - Urban Population

CONSISTENT_2000_2015 = ['SP.DYN.AMRT.MA',       # - Mortality rate, male
                        'SP.DYN.LE00.MA.IN',    # - Life expectancy, male
                        'SP.RUR.TOTL.ZS',       # - Percent Rural population
                        'SP.URB.TOTL.IN.ZS',    # - Percent Urban population
                        'EN.ATM.NOXE.EG.KT.CE', # - Energy no2 emissions
                        'SP.URB.TOTL',          # - Total Urban population
                        'SP.RUR.TOTL',          # - Total Rural population
                        'EG.ELC.RNEW.ZS',       # - Renewable electricity output
                        'SP.POP.GROW',          # - Percent population growth
                        'EN.ATM.METH.AG.KT.CE', # - Ag. methane emissions 
                        'SP.DYN.AMRT.FE',       # - Mortality rate, female
                        #'NY.ADJ.DMIN.CD',       # - Mineral Depletion
                        'EN.ATM.METH.EG.KT.CE', # - Energy methane emissions
                        'SP.RUR.TOTL.ZG',       # - Rural population growth
                        'SP.DYN.TFRT.IN',       # - Fertility rate
                        'EN.ATM.NOXE.AG.KT.CE', # - Ag. no2 emissions
                        'SP.DYN.LE00.IN',       # - Life expectancy
                        'SP.DYN.LE00.FE.IN',    # - Life expectancy, female
                        'SP.URB.GROW']          # - Urban population growth

def get_world():
    world = gpd.read_file('C:/Users/Sean/Documents/MATH_498/data/map/ne_110m_admin_0_countries.shp')
    fix = {'Norway': 'NOR', 'France': 'FRA', 'Northern Cyprus': 'CYP', 'Somaliland': 'SOM', 'Kosovo': 'RKS'}
    for row in world.index:
        if world.loc[row, 'NAME_LONG'] in fix:
            world.loc[row, 'ISO_A3'] = fix[world.loc[row, 'NAME_LONG']]
    return world

def iso_dict(start = '2000', stop = '2015'):
    df = pd.read_csv('C:/Users/Sean/Documents/MATH_498/data/world_bank/WDIData.csv')
    country_codes = set(get_world()['ISO_A3']) - set(('SSD',))
    groups = df.groupby('Country Code')
    iso_d = {}
    for code, group_df in tqdm(groups):
        if code in country_codes:
            group_df = group_df.set_index('Indicator Code')
            smooth_data(group_df, start, stop)
            iso_d[code] = group_df.loc[:, start: stop]
    return iso_d

def smooth_data(df, start = '2000', stop = '2015'):
    start, stop = int(start), int(stop)
    for row in df.index:
        for j in range(start, stop + 1):
            if pd.isna(df.loc[row, str(j)]):
                tot = n = 0
                prev, nex = df.loc[row, str(j-1)], df.loc[row, str(j+1)]
                for val in (prev, nex):
                    if not pd.isna(val):
                        tot += val
                        n += 1
                if n == 0:
                    continue
                else:
                    df.loc[row, str(j)] = tot / n

def smooth_column(df, feature):
    col = df[feature]
    jump = len(df.loc['USA', :])
    for i in range(len(col) // jump):
        base = i * jump
        new_values = []
        smoother = lambda c, i: c.iloc[i - 1: i + 2].sum() / 3
        for j in range(1, jump - 1):
            j = j + base
            new_values.append(smoother(col, j))
        col.iloc[base + 1: base + 1 + len(new_values)] = new_values

def select_features(iso_d, feats, start, stop):
    start, stop = int(start), int(stop)
    iso_codes = []
    years = []
    data = []
    for key, df in sorted(iso_d.items()):
        for yr in range(start, stop + 1):
            iso_codes.append(key)
            years.append(str(yr))
            data.append(df.loc[feats, str(yr)])
    return pd.DataFrame(data, index = [iso_codes, years], columns = feats)

def find_features(iso_d, exclude = None):
    feats = None
    for table in iso_d.values():
        if feats is None:
            feats = set(table.index)
        filled_feats = set(table.dropna(axis = 0).index)
        feats = feats.intersection(filled_feats)
        print(len(feats))
    if exclude:
        feats = {f for f in feats if f not in exclude}
    return feats

def fill_nas(iso_df):
    iso_df = iso_df.groupby(level = 0).backfill()
    return iso_df.fillna(iso_df.mean(axis = 0))

#=========================================================================
def main():
    iso_dfs = iso_dict('2000', '2017')
    print(len(iso_dfs))
    all_feats = NECESSARY_FEATS
    #all_feats = TARGET_FEATS + FEATURES + CONSISTENT_2000_2015
    data = select_features(iso_dfs, all_feats, '2000', '2017')
    filled_data = fill_nas(data)
    smooth_us = NECESSARY_FEATS
    for feat in smooth_us:
        smooth_column(filled_data, feat)
    filled_data['SM.POP.NETM'] = filled_data['SM.POP.NETM'] / filled_data['SP.POP.TOTL'] * 1000 #Turn Net Migration into a ratio
    filled_data.to_pickle('C:/Users/Sean/Documents/MATH_498/code/country_data.pkl')
    print(filled_data)

if __name__ == '__main__':
    main()