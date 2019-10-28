# basic libraries python
import os
import sys
import math
import itertools

# data handling
import pandas as pd # data processing, CSV file I/O, data manipulation as in SQL
import numpy as np # linear algebra

# graph plotting
import matplotlib.pyplot as plt # for the plot the graph 
# import seaborn as sns # it's more interactive
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.preprocessing import StandardScaler, MinMaxScaler # for normalization
from sklearn.pipeline import Pipeline # pipeline making
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score

# deep learning
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD 
from tensorflow.keras.callbacks import EarlyStopping

"""
Data Handler
"""
def load_data(dtype, path):
  if dtype == 'txt':
    df = pd.read_csv(path, sep=';', header=0, low_memory=False, infer_datetime_format=True, 
                    parse_dates={'datetime':[0,1]}, index_col=['datetime'])
  elif dtype == 'csv':
    df = pd.read_csv(path, parse_dates = True, index_col = 'datetime', low_memory = False)
  print('>>> dataset loaded successfully!')
  return df

def print_data(df, n=5, tail=False):
  print(df.head(n))
  if tail:
    print(df.tail(n))

def export_data(df, path):
  df.to_csv(path)
  print('>>> data {} exported succesfully!'.format(path))

def create_directory(path):
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)

"""
Get metadata
- row num
- col num etc
"""
def get_metadata(df):
  metadata = {
    'row_num' : df.shape[0],
    'col_num' : df.shape[1],
    'col' : df.columns.to_list()
  }
  print('[ METADATA ]')
  for key, val in metadata.items():
    print('{} => {}'.format(key, val))
  return metadata

"""
Data preparing
- Mark all missing value
- Convert datatype become numeric
"""
def prepare_data(df):
  # mark all missing values
  df.replace('?', np.nan, inplace=True)
  # make dataset numeric
  df_prepared = df.astype('float32')
  print('>>> dataset prepared successfully!')
  return df_prepared

"""
Data preprocessing
- Handle missing value
    - Method:
      - Custom (same time one day ago)
      - Fillna (mean, median, mode, bfill, ffill)
      - Interpolate (linear, polynomial)
- Resampling
    - Unit : D, h
    - Method: sum, mean, min, max
"""
def check_null_values(df):
  print('[ MISSING VALUE ]')
  print(df.isnull().sum())

def fill_missing_value(values):
  one_day = 60 * 24
  for row in range(values.shape[0]):
      for col in range(values.shape[1]):
          if np.isnan(values[row, col]):
            values[row, col] = values[row - one_day, col]

def handle_missing_value(df, method, attr=None):
  if method == 'custom':
    fill_missing_value(df.values)
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
  print('>>> missing value handled succesfully!')  
  return df

def resample_data(df, unit, method):
  if method == 'sum':
    df.resample(unit).sum()
  elif method == 'mean':
    df.resample(unit).mean().round(1)
  elif method == 'max':
    df.resample(unit).max()
  elif method == 'min':
    df.resample(unit).min()
  print('>>> dataset resampled successfully!')  
  return df

"""
Visualization/EDA
- Correlation
- Data
"""
# using matshow matplotlib
def visualize_correlation(df, method, title, eda_dir_path):
    path = eda_dir_path + '/correlation/'
    plt.matshow(df.corr(method=method), vmax=1, vmin=-1, cmap='PRGn')
    plt.title(title, size=12)
    plt.colorbar()
    # plt.show()
    plt.savefig(path + title + '.png')
    print(title + '.png saved!')

# using heatmap seaborn
# def visualize_correlation(df, method, title, eda_dir_path):
#     path = eda_dir_path + '/correlation/'
#     f, ax = plt.subplots(figsize=(15,15))
#     sns.heatmap(df.corr(method=method), annot=True, fmt='.3f', ax=ax)
#     ax.set_title(title)
#     # plt.show()
#     plt.savefig(path + title + '.png')
#     print(title + '.png saved!')

"""
Visualize Data
- Over day (all field)
  - plot
  - hist (distribution)
- Over year (certain field)
  - plot
  - hist (distribution)
- Over month
"""
def visualize_data(vtime, vtype, df, path, title, col=None, years=None, year=None):
  create_directory(path)
  fig, ax = plt.subplots(figsize=(18,18))
  if vtime == 'd':
    for i in range(len(df.columns)):
      plt.subplot(len(df.columns), 1, i+1)
      name = df.columns[i]
      if vtype == 'plot':
        plt.plot(df[name])
      elif vtype == 'hist':
        df[name].hist(bins=200)
      plt.title(name, y=0, loc = 'right')
      plt.yticks([])
  elif vtime == 'y':
    for i in range(len(years)):
      plt.subplot(len(years), 1, i+1)
      year = years[i]
      data = df[str(year)]
      if vtype == 'plot':
        plt.plot(data[col])
      elif vtype == 'hist':
        data[col].hist(bins = 200)
      plt.title(str(year), y = 0, loc = 'left')
  elif vtime == 'm':
    months = [i for i in range(1,13)]
    for i in range(len(months)):
      ax = plt.subplot(len(months), 1, i+1)
      month = year + '-' + str(months[i])
      try:
        data = df[month]
        data[col].hist(bins = 100)
        ax.set_xlim(0,5)
        plt.title(month, y = 0, loc = 'right')
      except:
        break
  # plt.show()
  ax.set_title(title)
  fig.tight_layout()
  fig.savefig(path + title + '.png')
  print(path + title + '.png saved!')

if __name__ == "__main__":
  # get command line arguments
  task = sys.argv[1]
  method = sys.argv[2]
  resample_arg = sys.argv[3].split(',')

  # constant path
  PATH = os.getcwd()
  DATASET_DIR_PATH = PATH + '/dataset/'
  DATASET_PATH = DATASET_DIR_PATH + 'household_power_consumption.txt'
  EXPORTED_DATASET_PATH = DATASET_DIR_PATH + 'power_consumption' + '_' + method + '.csv'
  EDA_DIR_PATH = PATH + '/eda/'

  print('=========================')
  print('{}'.format('Preparing and Preprocessing' if task == 1 else 'Exploratory Data Analysis (EDA)' if task == 2 else 'Deep Learning using LSTM'))
  print('=========================')
  print('Dataset path : {}'.format(DATASET_PATH))
  print('Missing value handling method : {}'.format(method))
  print('Resample : {}'.format(resample_arg))

  if task == 1:
    # load data
    df = load_data('txt', DATASET_PATH)
    print_data(df)    
    # get metadata
    metadata = get_metadata(df)
    # prepare_data
    df = prepare_data(df)
    # handle missing value
    check_null_values(df)
    if method == 'custom':
      df = handle_missing_value(df, method)
    else:
      for attr in metadata['col']:
        df = handle_missing_value(df, method, attr)
    check_null_values(df)   
    # export prepared data
    export_data(df, EXPORTED_DATASET_PATH)

  elif task == 2:
    data_name = 'Power Consumption ({})'.format(method)
    years = ['2007', '2008', '2009', '2010']

    ## load data
    df = load_data('csv', EXPORTED_DATASET_PATH)
    print_data(df)
    ## get metadata
    metadata = get_metadata(df)
    ## visualize correlation
    # visualize_correlation(df, 'spearman', data_name)
    ## resampling
    df_resampled = resample_data(df, 'D', 'sum')
    ## visualize data
    visualize_data('d', 'plot', df_resampled, EDA_DIR_PATH + 'plot-per-day/', data_name)
    for attr in metadata['col']:
      visualize_data('d', 'hist', df_resampled, EDA_DIR_PATH + 'hist-dist-per-day-' + attr + '/', data_name, attr)
      visualize_data('y', 'plot', df_resampled, EDA_DIR_PATH + 'plot-per-year-' + attr + '/', data_name, attr, years)
      visualize_data('y', 'hist', df_resampled, EDA_DIR_PATH + 'hist-dist-per-year-' + attr + '/', data_name, attr, years)
      # for year in years:
      #   visualize_data('m', 'hist', df_resampled, EDA_DIR_PATH + 'hist-dist-per-month-' + year + '-' + attr + '/', data_name, attr, None, year)

  elif task == 3:
    # Deep learning
    print('=========================')
    print('Deep Learning using LSTM')
    print('=========================')
    print('Dataset path : {}'.format(DATASET_PATH))
    print('Missing value handling method : {}'.format(method))

    # load data
    df = load_data('csv', EXPORTED_DATASET_PATH)
    print_data(df)
    # get metadata
    metadata = get_metadata(df)

    # resample data
    df_resampled = resample_data(df, resample_arg[0], resample_arg[1])
    print_data(df)
    # renew metadata
    metadata = get_metadata(df_resampled)
    


    



