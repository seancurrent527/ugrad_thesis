'''
Data cleaning script for WDIData.
'''
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

DISCOVER_FEATS = False

TARGET_FEATS = ['SM.POP.NETM', 'SP.DYN.CDRT.IN', 'SP.DYN.CBRT.IN', 'SP.POP.TOTL.NX']

FEATURES = {'EG.ELC.ACCS.ZS',       # - access to electricity (also has rural/urban)
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
            'EN.POP.DNST'}          # - population density
            #'SP.POP.TOTL',          # - total population
            #'EG.ELC.RNEW.ZS',       # - percent renewable energy
            #'SP.RUR.TOTL',          # - Rural Population
            #'SL.UEM.TOTL.NE.ZS',    # - total unemployment
            #'SP.URB.TOTL'}          # - Urban Population

CONSISTENT_2000_2015 = {'SP.DYN.AMRT.MA',       # - Mortality rate, male
                        'SP.DYN.LE00.MA.IN',    # - Life expectancy, male
                        'SP.RUR.TOTL.ZS',       # - Percent Rural population
                        'SP.POP.TOTL',          # - Total population
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
                        'SP.URB.GROW'}          # - Urban population growth


def iso_dict(start = '2000', stop = '2015'):
    df = pd.read_csv('C:/Users/Sean/Documents/Thesis/migration/data/world_bank/WDIData.csv')
    country_codes = set(gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))['iso_a3'])
    groups = df.groupby('Country Code')
    iso_d = {}
    for code, group_df in tqdm(groups):
        if code in country_codes:
            group_df = group_df.set_index('Indicator Code')
            smooth_data(group_df, start, stop)
            iso_d[code] = group_df.loc[:, start: stop]
            iso_d[code].loc['SP.POP.TOTL.NX'] = group_df.loc['SP.POP.TOTL', str(int(start) + 1): str(int(stop) + 1)]
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

def select_features(iso_d, feats):
    iso_codes = []
    years = []
    data = []
    for key, df in sorted(iso_d.items()):
        for yr in df.columns:
            iso_codes.append(key)
            years.append(yr)
            data.append(df.loc[feats, yr])
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

def main():
    iso_dfs = iso_dict('2000', '2017')
    print(len(iso_dfs))
    target_feats = TARGET_FEATS
    targets = select_features(iso_dfs, target_feats)
    filled_targets = fill_nas(targets)
    if DISCOVER_FEATS:
        feats = find_features(iso_dfs, exclude = target_feats)
        print(feats)
    else:
        feats = FEATURES | CONSISTENT_2000_2015
    features = select_features(iso_dfs, feats)
    filled_features = fill_nas(features)
    filled_targets.to_pickle('simple_targets.pkl')
    filled_features.to_pickle('simple_features.pkl')
    print(filled_targets)
    print(filled_features)

if __name__ == '__main__':
    main()