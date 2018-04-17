import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import ModelCheckpoint, CSVLogger

from lstm_vae import create_lstm_vae

# Select gpu
import sys
import os
gpu = sys.argv[-3]
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= "{}".format(gpu)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

dataname = sys.argv[-2] 

epochs = int(sys.argv[-1])

def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

def get_data():
    # read data from file

    if dataname == 'basque':
        y = np.array(pd.read_csv("data/basque-y.csv"))
        x = np.array(pd.read_csv("data/basque-x.csv"))
        timesteps = 14-1

    data = np.hstack((y,x))

    print('raw data shape', data.shape)     

    dataX = []
    for i in range(len(data) - timesteps - 1):
        x = data[i:(i+timesteps), :]
        dataX.append(x)
    return np.array(dataX)

if __name__ == "__main__":
    x = get_data()
    print('input shape', x.shape) 
    input_dim = x.shape[-1] 
    timesteps = x.shape[1] 
    batch_size = 1

    vae, enc, gen = create_lstm_vae(input_dim, 
        timesteps=timesteps, 
        batch_size=batch_size, 
        intermediate_dim=32,
        latent_dim=100,
        epsilon_std=1.)

    filepath="results/{}".format(dataname) + "/weights.{epoch:02d}-{val_loss:.3f}.hdf5"
    checkpointer = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, period=10, save_best_only=True)

    csv_logger = CSVLogger('results/{}/training_log_{}.csv'.format(dataname,dataname), separator=',', append=False)

    vae.fit(x, x, 
        epochs=epochs,
        verbose=1,
        callbacks=[checkpointer,csv_logger],
        validation_split=0.01)