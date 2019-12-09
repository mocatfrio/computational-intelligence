# import all libraries needed
import os, sys, math
from numpy import concatenate
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from preprocessing import preprocess
from preparing import prepare, normalize

def plot_hist_loss(history):
    pyplot.plot(history.history['loss'], label='Train Loss')
    pyplot.plot(history.history['val_loss'], label='Test Loss')
    pyplot.title('Model Loss')
    pyplot.ylabel('Loss')
    pyplot.xlabel('Epochs')
    pyplot.legend()
    pyplot.show()        

def plot_hist_acc(history):
    pyplot.plot(history.history['acc'], label='Train Acc')
    pyplot.plot(history.history['val_acc'], label='Test Acc')
    pyplot.title('Model Acc')
    pyplot.ylabel('Loss')
    pyplot.xlabel('Epochs')
    pyplot.legend()
    pyplot.show()  

# def visualize_test(start=0, end=0, vis_range=150):
#     aa = [x for x in range(vis_range)]
#     pyplot.figure(figsize=(8,4))
#     if start is 0:
#         pyplot.plot(aa, invert_actual[:end], marker='.', label="actual")
#         pyplot.plot(aa, invert_prediction[:end], 'r', label="prediction")
#     elif end is 0:
#         pyplot.plot(aa, invert_actual[start:], marker='.', label="actual")
#         pyplot.plot(aa, invert_prediction[start:], 'r', label="prediction")
#     else:
#         pyplot.plot(aa, invert_actual[start:end], marker='.', label="actual")
#         pyplot.plot(aa, invert_prediction[start:end], 'r', label="prediction")
#     pyplot.tight_layout()
#     sns.despine(top=True)
#     pyplot.subplots_adjust(left=0.07)
#     pyplot.ylabel('Concentration of PM 2.5', size=15)
#     pyplot.xlabel('Time step', size=15)
#     pyplot.legend(fontsize=15)
#     pyplot.show();

if __name__ == "__main__":
    # define path
    PATH = os.getcwd()
    DATASET_DIR_PATH = PATH + '/../../data/'
    DATASET_PATH = {
        'raw': DATASET_DIR_PATH + 'PRSA_data.csv',
        'preprocessed': DATASET_DIR_PATH + 'PRSA_data_preprocessed.csv'
    }

    # get command line arguments
    task = int(sys.argv[1])
    try:
        method = sys.argv[2]     
    except:
        method = 'zero'   
    
    if task == 1:
        preprocess(DATASET_PATH['raw'], DATASET_PATH['preprocessed'], method)
    elif task == 2:
        scaler, data = prepare(DATASET_PATH['preprocessed'])
        # reshape input to be 3D array [samples, timesteps, features] as an input to LSTM network
        data['train_input'] = data['train_input'].reshape((data['train_input'].shape[0], 1, data['train_input'].shape[1]))
        data['test_input'] = data['test_input'].reshape((data['test_input'].shape[0], 1, data['test_input'].shape[1]))
        # design model 
        model = Sequential()
        model.add(LSTM(100, input_shape=(data['train_input'].shape[1], data['train_input'].shape[2])))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit network
        history = model.fit(data['train_input'], data['train_output'], epochs=50, batch_size=72, validation_data=(data['test_input'], data['test_output']), callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)
        # print model
        print(model.summary())
        # plot history
        # plot_hist_acc(history)
        plot_hist_loss(history)

        # make prediction
        prediction = model.predict(data['test_input'])
        # invert scaling for forecast
        # reshape back to 2D
        data['test_input'] = data['test_input'].reshape((data['test_input'].shape[0], data['test_input'].shape[2]))
        invert_prediction = concatenate((prediction, data['test_input'][:, 1:]), axis=1)
        invert_prediction = scaler.inverse_transform(invert_prediction)
        invert_prediction = invert_prediction[:, 0]
        # invert scaling for actual
        # reshape back to 2D
        data['test_output'] = data['test_output'].reshape((len(data['test_output']), 1))
        invert_actual = concatenate((data['test_output'], data['test_input'][:, 1:]), axis=1)
        invert_actual = scaler.inverse_transform(invert_actual)
        invert_actual = invert_actual[:, 0]
        # calculate MAE, RMSE
        rmse = math.sqrt(mean_squared_error(invert_actual, invert_prediction))
        mae = mean_absolute_error(invert_actual, invert_prediction)
        print('Test RMSE: %.3f' % rmse)
        print('Test MAE: %.3f' % mae)
        # print('Train Accuration: {}'.format(history.history['acc']))
        # print('Test Accuration: {}'.format(history.history['val_acc']))


            


        

