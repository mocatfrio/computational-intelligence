"""
Preparing
- Normalize data
- Split data as test and train, input and output
"""
from pandas import read_csv, DataFrame
from numpy import reshape, array
from sklearn.preprocessing import MinMaxScaler
from preprocessing import get_metadata, print_data

def prepare(dataset_path):
    # get initial data and metadata
    df = load_data(dataset_path)
    metadata = get_metadata(df)
    print_data(df, metadata)
    # convert data values as array
    val = df['pollution'].values
    # reshape to change the dimension of data
    val = reshape(val, (-1, 1))
    # normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_val = scaler.fit_transform(val)
    # split data into train and test
    data = split_data(scaled_val)
    print_splitted_data(data)
    # split data into input and output
    data['train_in'], data['train_out'] = split_data(data['train'], False)
    data['test_in'], data['test_out'] = split_data(data['test'], False)
    print_splitted_data(data, True)
    print('Data Train Input')
    print(DataFrame(data['train_in']).head())
    print('Data Train Output')
    print(DataFrame(data['train_out']).head())
    return scaler, data 

# function to load preprocessed data
def load_data(path):
    return read_csv(path, header=0, index_col=0)

# function to split data
def split_data(data, is_horizontal=True, n_steps=24):
    if is_horizontal:
        # train size = 20%
        # test size = 80%
        train_size = int(len(data) * 0.20)
        test_size = len(data) - train_size
        print('Train size: ', train_size)
        print('Test size: ', test_size)
        # split data into train and test
        new_data = {
            'train': data[0:train_size,:],
            'test': data[train_size:len(data),:]
        }
        return new_data
    else:
        X, y = [], []
        for i in range(len(data)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the data
            if end_ix > len(data)-1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = data[i:end_ix, 0], data[end_ix, 0]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)

# function to print data
def print_splitted_data(data, is_shape=False):
    if is_shape:
        for key, val in data.items():
            print('{} => {}'.format(key, val.shape))
    else:
        for key, val in data.items():
            print('{} => {}'.format(key, val))