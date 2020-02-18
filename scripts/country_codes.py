import pandas as pd

def main():
    data = pd.read_html('./data/country_codes.html')
    data[0].iloc[:, 1:-1].to_csv('data/country_codes.csv')

if __name__ == '__main__':
    main()