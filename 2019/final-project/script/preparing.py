"""
Preparing
- Normalize data
- Split data as test and train, input and output
"""

from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from preprocessing import get_metadata, print_data

def prepare(dataset_path):
    # get initial data and metadata
    df = load_data(dataset_path)
    metadata = get_metadata(df)
    print_data(df, metadata)
    # convert data values as array
    val = df.values
    # normalize data
    scaler, scaled_val = normalize(val)
    # reframe as supervised learning
    reframed = series_to_supervised(scaled_val, 1, 1)
    # drop all the unneeded columns 
    reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
    print_data(reframed)
    # split data test and train
    data = split_data(reframed)
    print('Number of train data: {}'.format(len(data['train'])))
    print('Number of test data: {}'.format(len(data['test'])))
    # split data input and output
    data = split_data(data, False)
    return scaler, data 

def load_data(path):
    return read_csv(path, header=0, index_col=0)

def normalize(val):
    # convert wind direction (SE and NW) into integer
    encoder = LabelEncoder()
    val[:, 4] = encoder.fit_transform(val[:, 4])
    # convert all datatype into float
    val = val.astype('float32')
    # normalize features with a range of 0 to 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_val = scaler.fit_transform(val)
    return scaler, scaled_val

"""
Split Data
"""
# by default, using first 1 year as train dataset
def split_data(data, is_horizontal=True, train_size=365*24):
    if is_horizontal:
        new_data = {}
        # split into train and test datasets 
        val = data.values
        new_data['train'] = val[:train_size, :]
        new_data['test'] = val[train_size:, :]
        return new_data
    else:
        # split into input and output
        data['train_input'], data['train_output'] = data['train'][:, :-1], data['train'][:, -1]
        data['test_input'], data['test_output'] = data['test'][:, :-1], data['test'][:, -1]
        return data

"""
Convert Timeseries to Supervised 
"""
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    if type(data) is list:
        n_vars = 1
    else:
        n_vars = data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ..., t-1) 
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]    
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg