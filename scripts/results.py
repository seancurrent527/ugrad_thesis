import numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd, geopandas as gpd
from gravity_regression import iso_to_continent
from model_plot import get_world, _parse_args
from sklearn.metrics import r2_score

def summary(vals):
    # return {'min': np.min(vals),
    #         'max': np.max(vals),
    #         'mean': np.mean(vals),
    #         'std': np.std(vals),
    #         'median': np.median(vals),
    #         'n': len(vals)}
    stats = [np.min(vals), np.max(vals), np.mean(vals),
             np.std(vals), np.median(vals)]
    return ' & '.join(map(lambda x: str(round(x, 3)), stats))




def main():
    args = _parse_args()
    modtype = ('subset_' * bool(args.continent)) + 'state' + ('less' * (1 - args.state_model))
    targets = pd.read_pickle('C:/Users/Sean/Documents/MATH_498/code/country_data.pkl')
    year_data_p = pd.read_csv('C:/Users/Sean/Documents/MATH_498/code/generated_data/2000_' + modtype + '_p_out.csv')
    year_data_m = pd.read_csv('C:/Users/Sean/Documents/MATH_498/code/generated_data/2000_' + modtype + '_m_out.csv')
    year_data_d = pd.read_csv('C:/Users/Sean/Documents/MATH_498/code/generated_data/2000_' + modtype + '_d_out.csv')
    year_data_b = pd.read_csv('C:/Users/Sean/Documents/MATH_498/code/generated_data/2000_' + modtype + '_b_out.csv')
    p_targs = pd.DataFrame(data = np.array([df['SP.POP.TOTL'].values for _, df in targets.groupby(level = 0)]).T,
                           columns = year_data_p.columns)
    m_targs = pd.DataFrame(data = np.array([df['SM.POP.NETM'].values for _, df in targets.groupby(level = 0)]).T,
                           columns = year_data_m.columns)
    d_targs = pd.DataFrame(data = np.array([df['SP.DYN.CDRT.IN'].values for _, df in targets.groupby(level = 0)]).T,
                           columns = year_data_d.columns)
    b_targs = pd.DataFrame(data = np.array([df['SP.DYN.CBRT.IN'].values for _, df in targets.groupby(level = 0)]).T,
                           columns = year_data_b.columns)

    conts = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
    conts_ = {c:c for c in conts}
    conts_['World'] = 'World'
    continents = iso_to_continent()
    for targs, preds, tp in zip([p_targs, m_targs, d_targs, b_targs],
                                [year_data_p, year_data_m, year_data_d, year_data_b],
                                ['population', 'migration', 'death rate', 'birth rate']):
        if tp == 'death rate':
            for c in conts + ['World']:
                print(conts_[c] + r' \\')
            print()
            conts_ = {c:c for c in conts}
            conts_['World'] = 'World'
        #print(tp)
        for c in conts:
            countries = list(filter(lambda x: x in targs.columns, continents.index[continents == c]))
            vals = []
            for i in countries:
                vals.append(np.corrcoef(targs[i].iloc[:15], preds[i].iloc[:15])[0, 1])
                #vals.append(r2_score(targs[i].iloc[:15], preds[i].iloc[:15]))
            conts_[c] += ' & ' + summary(vals)
        vals = []
        for i in targs.columns:
            vals.append(np.corrcoef(targs[i].iloc[:15], preds[i].iloc[:15])[0, 1])
            #vals.append(r2_score(targs[i].iloc[:15], preds[i].iloc[:15]))
        conts_['World'] += ' & ' + summary(vals)
    for c in conts + ['World']:
        print(conts_[c] + r' \\')
        

if __name__ == '__main__':
    main()