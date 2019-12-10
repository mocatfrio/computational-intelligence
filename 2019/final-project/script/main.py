# import all libraries needed
import os, sys, math
from datetime import datetime
from preprocessing import preprocess
from preparing import prepare
from forecasting import forecast

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
    if task == 2:
        method = sys.argv[2]
    # do the tasks
    if task == 1:
        preprocess(DATASET_PATH['raw'], DATASET_PATH['preprocessed'])
    elif task == 2:
        scaler, data = prepare(DATASET_PATH['preprocessed'])
        forecast(method, data, scaler)        

            


        

