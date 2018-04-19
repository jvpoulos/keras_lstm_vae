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
        n_post = 1 
        n_pre = 14-1
        seq_len = 43

    if dataname == 'california':
        n_post  = 1 
        n_pre =  19-1 
        seq_len = 31

    if dataname == 'germany':
        n_post  = 1 
        n_pre =  30-1 
        seq_len = 44   
        
    y = np.array(pd.read_csv("data/{}-y.csv".format(dataname)))
    x = np.array(pd.read_csv("data/{}-x.csv".format(dataname)))    

    data = np.hstack((y,x))

    print('raw data shape', data.shape)     

    dataX =[]
    for i in range(seq_len - n_pre - n_post):
        dataX.append(data[i:i+n_pre])
    return np.array(dataX), n_pre, n_post

if __name__ == "__main__":
    x, n_pre, n_post = get_data() 
    nb_features = x.shape[2]
    batch_size = 1

    vae, enc, gen = create_lstm_vae(nb_features, 
        n_pre=n_pre, 
        n_post=n_post,
        batch_size=batch_size, 
        intermediate_dim=32,
        latent_dim=100,
        initialization = 'glorot_normal',
        activation = 'linear',
        lr = 0.001,
        penalty=0.001,
        dropout=0.5,
        epsilon_std=1.)

    filepath="results/{}".format(dataname) + "/weights.{epoch:02d}-{val_loss:.3f}.hdf5"
    checkpointer = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, period=100, save_best_only=True)

    csv_logger = CSVLogger('results/{}/training_log_{}.csv'.format(dataname,dataname), separator=',', append=False)

    vae.fit(x, x, 
        epochs=epochs,
        verbose=1,
        callbacks=[checkpointer,csv_logger],
        validation_split=0.01)