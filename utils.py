import pandas as pd
from model_config import __DATA_COLUMN__, __COLS_TO_KEEP__


def get_csv_given_date(csv_path, start_date, end_date):
    df = pd.read_csv(csv_path)
        
    df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%b-%y')
    starting_date = pd.to_datetime(start_date, format='%d-%m-%Y')
    ending_date = pd.to_datetime(end_date, format='%d-%m-%Y')
    
    filtered_df = df[(df['DATE'] >= starting_date) & (df['DATE'] <= ending_date)]
    
    return filtered_df[__COLS_TO_KEEP__].to_string(index=False)


    