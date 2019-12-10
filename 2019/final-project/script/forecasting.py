from numpy import sqrt, reshape
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

def forecast(model_type, data, scaler):
    # adjust dimension of data 
    data = reshape_data(model_type, data)
    # define model
    model = define_model(model_type, data)
    # get history of training
    history = training(data, model)
    # print model
    print(model.summary())
    # evaluate performance
    prediction = evaluate(data, model, scaler)
    show_history(history)
    show_prediction(data, prediction)

def reshape_data(model_type, data, n_features=1):
    if model_type == 'lstm':
        # reshape input menjadi 3D array [samples, time steps, features]
        data['train_in'] = reshape(data['train_in'], (data['train_in'].shape[0], n_features, data['train_in'] .shape[1]))
        data['test_in']  = reshape(data['test_in'], (data['test_in'].shape[0], n_features, data['test_in'].shape[1]))
    elif model_type == 'cnn':
        # reshape input menjadi 3D array : [samples, time steps, features]
        data['train_in'] = reshape(data['train_in'], (data['train_in'].shape[0], data['train_in'] .shape[1], n_features))
        data['test_in']  = reshape(data['test_in'], (data['test_in'].shape[0], data['test_in'].shape[1], n_features))
    return data

def define_model(model_type, data):
    model = Sequential()
    if model_type == 'lstm':
        model.add(LSTM(100, input_shape=(data['train_in'].shape[1], data['train_in'].shape[2])))
    elif model_type == 'cnn':
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(24, 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def training(data, model, epoch=50, batch_size=70):
    # train data and recorded in history
    history = model.fit(data['train_in'], data['train_out'], epochs=epoch, batch_size=batch_size, validation_data=(data['test_in'], data['test_out']), verbose=1, shuffle=False)
    return history

def evaluate(data, model, scaler):
    # predicting
    prediction = {
        'train_predict': model.predict(data['train_in']),
        'test_predict': model.predict(data['test_in'])
    }
    # invert prediction
    prediction['train_predict'] = scaler.inverse_transform(prediction['train_predict'])
    data['train_out'] = scaler.inverse_transform([data['train_out']])
    prediction['test_predict'] = scaler.inverse_transform(prediction['test_predict'])
    data['test_out'] = scaler.inverse_transform([data['test_out']])
    # evaluate performance
    print('Train MAE:', mean_absolute_error(data['train_out'][0], prediction['train_predict'][:,0]))
    print('Train RMSE:',sqrt(mean_squared_error(data['train_out'][0], prediction['train_predict'][:,0])))
    print('Test MAE:', mean_absolute_error(data['test_out'][0], prediction['test_predict'][:,0]))
    print('Test RMSE:',sqrt(mean_squared_error(data['test_out'][0], prediction['test_predict'][:,0])))
    return prediction

def show_history(history):
    # plotting loss
    plt.figure(figsize=(20,8))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')
    plt.show()

def show_prediction(data, prediction):
    # show actuals and predictions on training and testing data in the first week
    timesteps = 24*7 
    aa = [x for x in range(timesteps)]

    plt.figure(figsize=(20,8))
    plt.plot(aa, data['train_out'][0][:timesteps], marker='.', label="actual")
    plt.plot(aa, prediction['train_predict'][:,0][:timesteps], 'r', label="prediction")
    plt.tight_layout()
    plt.subplots_adjust(left=0.07)
    plt.title('Training')
    plt.ylabel('PM 2.5 Concentration', size=15)
    plt.xlabel('Time step', size=15)
    plt.legend(fontsize=15)
    plt.show()

    plt.figure(figsize=(20,8))
    plt.plot(aa, data['test_out'][0][:timesteps], marker='.', label="actual")
    plt.plot(aa, prediction['test_predict'][:,0][:timesteps], 'r', label="prediction")
    plt.tight_layout()
    plt.subplots_adjust(left=0.07)
    plt.title('Testing')
    plt.ylabel('PM 2.5 Concentration', size=15)
    plt.xlabel('Time step', size=15)
    plt.legend(fontsize=15)
    plt.show()