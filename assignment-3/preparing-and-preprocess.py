# Import package
import os
import pandas as pd # data processing, CSV file I/O, data manipulation as in SQL
import numpy as np # linear algebra
import matplotlib.pyplot as plt # for the plot the graph 
import math

def load_data(path):
  df = pd.read_csv(path, sep=';', header=0, low_memory=False, infer_datetime_format=True, 
                  parse_dates={'datetime':[0,1]}, index_col=['datetime'])
  return df

def view_data(df, n=5):
  print(df.head(n))
  # print(df.tail(n))

"""
get metadata
such as row num, col num etc
"""
def get_metadata(df):
  metadata = {
    'row_num' : df.shape[0],
    'col_num' : df.shape[1],
    'col' : df.columns.to_list(),
    'null_values' : []
  }
  for i in df.isnull().sum():
    metadata['null_values'].append(i)
  for key, val in metadata.items():
    print('{} => {}'.format(key, val))
  return metadata

"""
data preparing
- mark all missing value
- convert datatype become numeric
"""
def preparing_data(df):
  # mark all missing values
  df.replace('?', np.nan, inplace=True)
  # make dataset numeric
  df_prepared = df.astype('float32')
  return df_prepared

"""
data preprocessing
- handle missing value
  method:
    - custom (same time one day ago)
    - fillna (mean, median, mode, bfill, ffill)
    - interpolate (linear, polynomial)
- 
"""
def fill_missing_value(values):
  one_day = 60 * 24
  for row in range(values.shape[0]):
      for col in range(values.shape[1]):
          if np.isnan(values[row, col]):
            values[row, col] = values[row - one_day, col]

def handle_missing_value(df, method, attr):
  if method == 'mean':
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

def check_null_values(df):
  print(df.isnull().sum())

def export_data(df, path):
  df.to_csv(path)
  print('data {} exported succesfully!'.format(path))

if __name__ == "__main__":
  DATASET_PATH = os.getcwd() + '/dataset/'
  methods = ['custom', 'mean', 'median', 'mode', 'bfill', 'ffill', 'linear', 'polynomial']

  for method in methods:
    print('=========================')
    print(method)
    # load data
    df = load_data(DATASET_PATH + 'household_power_consumption.txt')
    # get metadata
    metadata = get_metadata(df)
    # preparing_data
    df = preparing_data(df)

    # fill missing value
    check_null_values(df)

    if method == 'custom':
      fill_missing_value(df.values)
    else:
      for attr in metadata['col']:
        df = handle_missing_value(df, method, attr)
    
    check_null_values(df)   

    # export prepared data
    export_data(df, DATASET_PATH + 'household_power_consumption' + '_' + method + '.csv')