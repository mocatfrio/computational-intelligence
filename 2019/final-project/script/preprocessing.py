"""
Preprocessing Data
- Fill missing value
"""
from pandas import read_csv
from datetime import datetime

def preprocess(dataset_path, export_path):
    # get initial data and metadata
    df = load_data(dataset_path)
    metadata = get_metadata(df)
    print_data(df, metadata)
    # get attr that has null value based on the metadata
    nullAttr = []
    for key, val in metadata['null_value'].items():
        if val > 0:
            nullAttr.append(key)
    # fill missing values with zero
    for attr in nullAttr:
        df[attr].fillna(0, inplace=True)
    # drop the first 24 hours or 1 day because the pollution table is 0
    df = df[24:]
    # get new metadata
    metadata = get_metadata(df)
    print_data(df, metadata)
    # export data
    export_data(df, export_path)

"""
Data Handling
"""
# function to load data
def load_data(path):
    # read all column names 
    cols = list(read_csv(path, nrows =1))
    df = read_csv(path, sep=',', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':['year', 'month', 'day', 'hour']}, date_parser=custom_parser, index_col=['datetime'], usecols=[i for i in cols if i != 'No'])
    # specify new column names
    df.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    return df

# function to export data
def export_data(df, path):
  df.to_csv(path)
  print('>>> data exported succesfully!')

# function to get metadata
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

# function to parse datetime, specifically for this case 
def custom_parser(year, month, day, hour):
    date_string = year + ' ' + month + ' ' + day + ' ' + hour
    return datetime.strptime(date_string, '%Y %m %d %H')

# function to print metadata
def print_data(df, metadata=None):
    print('=====================================')
    print('[ DATA HEAD ]')
    print(df.head(5))
    if metadata:
        print('[ METADATA ]')
        for key, val in metadata.items():
            print('{} => {}'.format(key, val))
    print('=====================================')