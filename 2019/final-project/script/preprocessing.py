"""
Preprocessing Data
- Fill missing value
"""
import pandas as pd
import numpy as np
import math
from datetime import datetime

def preprocess(dataset_path, export_path, fill_method='zero'):
    # get initial data and metadata
    df = load_data(dataset_path)
    metadata = get_metadata(df)
    print_data(df, metadata)
    # asumsinya dtypesnya udh bener
    # ambil attr yg punya null value berdasarkan metadata
    nullAttr = []
    for key, val in metadata['null_value'].items():
        if val > 0:
            nullAttr.append(key)
    # fill missing value
    for attr in nullAttr:
        df = fill_missing_value(df, fill_method, attr)
    # drop the first 24 hours or 1 day because the pollution table is 0
    df = df[24:]

    metadata = get_metadata(df)
    print_data(df, metadata)

    export_data(df, export_path)

def load_data(path):
    # read all column names 
    cols = list(pd.read_csv(path, nrows =1))
    df = pd.read_csv(path, sep=',', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':['year', 'month', 'day', 'hour']}, date_parser=custom_parser, index_col=['datetime'], usecols=[i for i in cols if i != 'No'])
    # specify new column names
    df.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    return df

def export_data(df, path):
  df.to_csv(path)
  print('>>> data exported succesfully!')

def get_metadata(df):
  return {
    'row_num' : df.shape[0],
    'col_num' : df.shape[1],
    'attr' : df.columns.to_list(),
    'timeseries_start': df.index.min(),
    'timeseries_end': df.index.max(),
    'null_value': df.isnull().sum().to_dict(),
    'dtypes': df.dtypes.to_dict()
  }

# parse datetime
def custom_parser(year, month, day, hour):
    date_string = year + ' ' + month + ' ' + day + ' ' + hour
    return datetime.strptime(date_string, '%Y %m %d %H')

"""
Handle Missing Value 
"""
def custom_fill(values):
  one_day = 60 * 24
  for row in range(values.shape[0]):
      for col in range(values.shape[1]):
          if np.isnan(values[row, col]):
            values[row, col] = values[row - one_day, col]

def fill_missing_value(df, method, attr=None):
    if method == 'custom':
        custom_fill(df.values)
    elif method == 'zero':
        df[attr].fillna(0, inplace=True)
    elif method == 'mean':
        try:
            if not math.isnan(df[attr].mean()):
                df[attr].fillna(df[attr].mean().round(2), inplace=True)
        except RuntimeWarning:
            pass
    elif method == 'median':
        df[attr].fillna(df[attr].median(), inplace=True)
    elif method == 'mode':
        if not df[attr].mode().empty:
            df[attr].fillna(df[attr].mode()[0], inplace=True)
    elif method == 'bfill' or method == 'ffill':
        df[attr].fillna(method=method, inplace=True)
    elif method == 'linear':
        df[attr].interpolate(method=method, limit_direction='forward', axis=0, inplace=True)
        df[attr] = df[attr].round(2)
    elif method == 'polynomial':
        try:
            df[attr].interpolate(method=method, order=2, inplace=True)
            df[attr] = df[attr].round(2)
        except FutureWarning:
            pass
    return df

def print_data(df, metadata=None):
    print('=====================================')
    print('[ DATA HEAD ]')
    print(df.head(5))
    print('[ DATA TAIL ]')
    print(df.tail(5))
    if metadata:
        print('[ METADATA ]')
        for key, val in metadata.items():
            print('{} => {}'.format(key, val))
    print('=====================================')