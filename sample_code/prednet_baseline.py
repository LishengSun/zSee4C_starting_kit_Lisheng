'''
Evaluate trained PredNet on KITTI sequences.
Calculates mean-squared error and plots predictions.
'''

import os
import numpy as np
#from six.moves import cPickle
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec

#from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
from prednet_utils import SequenceGeneratorChalearn
#from chalearn_analyse import analyse_results
#from chalearn_settings import * # Isabelle: I got rid of this file
from data_io import vprint


np.random.seed(123)
#from six.moves import cPickle

#from keras import backend as K
#from keras.models import Model
#from keras.layers import Input, Dense, Flatten
#from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
#from keras.optimizers import Adam
#import tensorflow as tf


import sys
sys.path.append('code')

# ------ add by julio -----------
import h5py
from data_manager import DataManager
#import cv2
#from time import time
# ------ add by julio -----------

class prednet_baseline():
    def __init__(self, hyper_param, path="", verbose=False):
        for key in hyper_param:
            setattr(self, key, hyper_param[key])
        self.model_dir = path
        # We could set other hyper_parameters from outside, for now they are fixed
        self.verbose=verbose
        vprint(self.verbose, "--> Version = " + self.version)
        vprint(self.verbose, "--> Cache dir = " + self.cache_dir)
        vprint(self.verbose, "--> Model dir = " + self.model_dir)
        vprint(self.verbose, "--> Hyper_parameters : ")
        vprint(self.verbose, hyper_param)
        self.WEIGHTS_DIR = self.cache_dir
        self.DATA_DIR = self.cache_dir
        self.nt = self.S_HELD + self.S_PRED
        self.weights_file = os.path.join(path, self.WEIGHTS_DIR, self.model_name + '_weights.hdf5')
        self.json_file = os.path.join(self.WEIGHTS_DIR, self.model_name + '_model.json')
        print(self.json_file)

        # Load or initialize trained model
        try:
            with open(self.json_file, "r") as f:
                vprint(self.verbose, "Loading trained prednet model")
                json_string = f.read()
                f.close()
                self.train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
                self.train_model.load_weights(self.weights_file)
        except:
            vprint(self.verbose, "No pretrained prednet model, initializing")
            self.init()

        # Create testing model (to output predictions)
        vprint(self.verbose, "Creating test prednet model")
        layer_config = self.train_model.layers[1].get_config()
        layer_config['output_mode'] = 'prediction'
        self.dim_ordering = layer_config['dim_ordering']
        test_prednet = PredNet(weights=self.train_model.layers[1].get_weights(), **layer_config)
        input_shape = list(self.train_model.layers[0].batch_input_shape[1:])
        input_shape[0] = self.nt
        inputs = Input(shape=tuple(input_shape))
        predictions = test_prednet(inputs)
        self.test_model = Model(input=inputs, output=predictions)
##-------------------------------------------

    def generate_train_data(self, data_dir):
       ''' This function rearrange the hackathon data to the one used by the prednet (first, for train data)'''
       data_dir = os.path.join(data_dir, 'train/Xm1')
       if not os.path.exists(data_dir):
          print "train/Xm1 folder not found"
          exit()
    
       new_data = np.zeros((self.TRAIN_SAMPLES[0]*3,101+8,1,32,32))
       print "Generating train data..."
    
       # for each video
       cont = -1
       for i in range(1,self.TRAIN_SAMPLES[0]+1):
          print i
          filename = 'X' + str(i) + '.h5'
          f = h5py.File(os.path.join(data_dir, filename), 'r')
          frames = f['X']['value']['X']['value'][:]
    
          # clip 1
          cont = cont+1
          cont_aux = 0
          for j in range(0,101+8):
             new_data[cont,cont_aux,0,0:32,0:32] = frames[j,0:32,0:32]
             cont_aux = cont_aux+1
    
          cont = cont+1
          # clip 2
          cont_aux = 0
          delta = 8
          for j in range(0+delta,101+8+delta):
             new_data[cont,cont_aux,0,0:32,0:32] = frames[j,0:32,0:32]
             cont_aux = cont_aux+1
    
          cont = cont+1
          # clip 3
          cont_aux = 0
          delta = 16
          for j in range(0+delta,101+8+delta):
             new_data[cont,cont_aux,0,0:32,0:32] = frames[j,0:32,0:32]
             cont_aux = cont_aux+1
    
    
       # save the rearranged data (train set)
       if not os.path.exists(self.DATA_DIR):
          os.makedirs(self.DATA_DIR)
       np.save(os.path.join(self.DATA_DIR, self.version + '_train.npy'), new_data)
          
    
    
    #-------------------------------------------
    # this function rearrange the hackathon data to the one used by the prednet (for validation data)
    def generate_valid_data(self, data_dir):
       data_dir = os.path.join(data_dir, 'train/Xm1')
       if not os.path.exists(data_dir):
          print "train/Xm1 folder not found"
          exit()
    
       print "Generating validation data..."
       valid_size = self.TRAIN_SAMPLES[1]-self.TRAIN_SAMPLES[0]
       new_data = np.zeros((valid_size*3,101+8,1,32,32))
    
       # for each video
       cont = -1
       for i in range(self.TRAIN_SAMPLES[0]+1,self.TRAIN_SAMPLES[1]+1):
          print i
          filename = 'X' + str(i) + '.h5'
          f = h5py.File(os.path.join(data_dir, filename), 'r')
          frames = f['X']['value']['X']['value'][:]
    
          # clip 1
          cont = cont+1
          cont_aux = 0
          for j in range(0,101+8):
             new_data[cont,cont_aux,0,0:32,0:32] = frames[j,0:32,0:32]
             cont_aux = cont_aux+1
    
          cont = cont+1
          # clip 2
          cont_aux = 0
          delta = 8
          for j in range(0+delta,101+8+delta):
             new_data[cont,cont_aux,0,0:32,0:32] = frames[j,0:32,0:32]
             cont_aux = cont_aux+1
    
          cont = cont+1
          # clip 3
          cont_aux = 0
          delta = 16
          for j in range(0+delta,101+8+delta):
             new_data[cont,cont_aux,0,0:32,0:32] = frames[j,0:32,0:32]
             cont_aux = cont_aux+1
    
    
    
       # save the rearranged data (validation set)
       if not os.path.exists(self.DATA_DIR):
          os.makedirs(self.DATA_DIR)
       np.save(os.path.join(self.DATA_DIR, self.version + '_valid.npy'), new_data)
    
    



    def generate_test_file(self, data_path,clip_id,id_aux):
       # aux data file 
       data = np.zeros((1,109,1,32,32))
       NEW_DATA_DIR = os.path.join(data_path, 'adapt/')
    
       # if it is the first clip (only one with 101 frames in the dataset)
       if id_aux==0:
          #print "file %s :101" %id_aux
          f = h5py.File(os.path.join(NEW_DATA_DIR,'X' + str(clip_id) + '.h5'), 'r')
          frames = f['X']['value']['X']['value'][:]
          #print frames.shape
          for j in range(0,101):
             data[0,j,0,0:32,0:32] = frames[j,0:32,0:32]
    
       # if it is the '8 length' clip
       if id_aux==1 or id_aux==2:
          #print "file %s :8" %id_aux
          data = np.load(os.path.join(self.DATA_DIR, self.version + '_test.npy'), 'r+')
          f = h5py.File(os.path.join(NEW_DATA_DIR,'X' + str(clip_id) + '.h5'), 'r')
          frames = f['X']['value']['X']['value'][:]
          #print frames.shape
          # shift the data first
          for j in range(8,101):
             data[0,j-8,0,0:32,0:32] = data[0,j,0,0:32,0:32]
          # append the new data
          for j in range(0,8):
             data[0,93+j,0,0:32,0:32] = frames[j,0:32,0:32]
    
       # if it is the '109 length' clip
       if id_aux==3:
          #print "file %s :109" %id_aux
          data = np.zeros((1,109,1,32,32))
          # loading previous file (again)
          f = h5py.File(os.path.join(NEW_DATA_DIR,'X' + str(clip_id) + '.h5'), 'r')
          frames = f['X']['value']['X']['value'][:]
          #print frames.shape
          # getting the 101 frames of the next clip (ignoring the first 8 frames of the previous one)
          for j in range(8,109):
             data[0,j-8,0,0:32,0:32] = frames[j,0:32,0:32]
    
       for j in range(101,109):
          data[0,j,0,0:32,0:32] = 0
    
       # removing previous saved file
       if os.path.exists(os.path.join(self.DATA_DIR, self.version + '_test.npy')):
          os.system('rm ' +os.path.join(self.DATA_DIR, self.version + '_test.npy'))
       # saving the new data
       np.save(os.path.join(self.DATA_DIR,'faces_32_1_test.npy'), data)
    
    
    
    def predict(self, data_path, OUTPUT_DATA, clip_id):
    
       # creating the idx to shift the data (0, 1, 2, 3; 1, 2, 3; 1, 2, 3; ...; 1, 2, 3)
       if clip_id<4:
          id_aux = clip_id
       else:
          if clip_id%3==0:
             id_aux = 3
          else:
             id_aux = clip_id%3
    
       # from the h5py file, generate the npy file in the NEW_DATA_DIR_AUX directory
       self.generate_test_file(data_path,clip_id,id_aux)
       test_file = os.path.join(self.DATA_DIR, self.version + '_test.npy')
    
       # Prepare ground truth and predictions
       X_test = SequenceGeneratorChalearn(test_file, self.nt, unique_start='unique', dim_ordering=self.dim_ordering).create_all()
       X_hat = np.copy(X_test)
       for i in range(self.S_HELD, self.S_HELD+self.S_PRED):
          X_hat[:, i] = self.test_model.predict(X_hat, self.test_batch_size)[:, i]
    
       # Reorder gt and predictions dimensions if using theano dimension ordering
       if self.dim_ordering == 'th':
          #X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
          X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))
    
       
       # save only predicted ones (las 8) and not the whole clip (109) (1, 109, 32, 32, 1) -> (1, 8, 32, 32, 1)
       predicted = np.zeros((8,32,32))
       for i in range(0,8):
          predicted[i,0:32,0:32] = X_hat[0,101+i,0:32,0:32,0]
    
       Dout = DataManager(datatype="output", verbose=True)
       Dout.X = predicted
       Dout.saveData('Y' + str(clip_id), data_dir=OUTPUT_DATA, format="h5")
    
       #X_hat = np.transpose(X_hat, (0, 1, 4, 2, 3))
       
       # visualize the output
    #  if ENABLE_IMG_GENERATION == True:
    #     X_hat = np.transpose(X_hat, (0, 1, 4, 2, 3))
    #     generate_seq(X_hat,clip_id)
    
    def init(self):
       '''
       Initialize PredNet on KITTI sequences. (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
       If debug = True, only initialize, perform no training.
       '''
   
       # Model parameters
       stack_sizes = (self.input_shape[0], self.stack_size[0], self.stack_size[1],self.stack_size[2])
       R_stack_sizes = stack_sizes
       layer_loss_weights = np.array(self.layer_loss_weights)
       layer_loss_weights = np.expand_dims(self.layer_loss_weights, 1)
       time_loss_weights = 1./ (self.nt - 1) * np.ones((self.nt,1))
       time_loss_weights[0] = 0
    
       prednet = PredNet(stack_sizes, R_stack_sizes,
                      self.A_filt_sizes, self.Ahat_filt_sizes, self.R_filt_sizes,
                      output_mode='error', return_sequences=True)
    
       inputs = Input(shape=(self.nt,) + self.input_shape)
       errors = prednet(inputs)  # errors will be (self.train_batch_size, self.nt, nb_layers)
       errors_by_time = TimeDistributed(Dense(1, weights=[layer_loss_weights, np.zeros(1)], trainable=False), trainable=False)(errors)  # calculate weighted error by layer
       errors_by_time = Flatten()(errors_by_time)  # will be (self.train_batch_size, self.nt)
       final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
       self.train_model = Model(input=inputs, output=final_errors)
       self.train_model.compile(loss=self.loss, optimizer=self.optimizer)


    def fit(self, data_path, save_model = True):
       '''
       Train PredNet on KITTI sequences. (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
       '''    
       # Prepare data
       self.generate_train_data(data_path)
       self.generate_valid_data(data_path)
    
       # Data files
       train_file = os.path.join(self.DATA_DIR, self.version + '_train.npy')
       val_file = os.path.join(self.DATA_DIR, self.version + '_valid.npy')

       # Parameters      
       N_seq_val = self.TRAIN_SAMPLES[1]-self.TRAIN_SAMPLES[0]

       train_generator = SequenceGeneratorChalearn(train_file, self.nt, batch_size=self.train_batch_size, shuffle=True)
       val_generator = SequenceGeneratorChalearn(val_file, self.nt, batch_size=self.train_batch_size, N_seq=N_seq_val)
    
       lr_schedule = lambda epoch: self.start_lr if epoch < 75 else self.end_lr,    # start with lr large and then drop to smaller value after 75 epochs
       callbacks = [LearningRateScheduler(lr_schedule)]
       if save_model:
           if not os.path.exists(self.WEIGHTS_DIR): os.mkdir(self.WEIGHTS_DIR)
           callbacks.append(ModelCheckpoint(filepath=self.weights_file, monitor='val_loss', save_best_only=True))
    
       self.history = self.train_model.fit_generator(train_generator, self.samples_per_epoch, self.nb_epoch, callbacks=callbacks,
                        validation_data=val_generator, nb_val_samples=N_seq_val)
    
       if save_model:
           json_string = self.train_model.to_json()
           with open(self.json_file, "w") as f:
               f.write(json_string)
    
