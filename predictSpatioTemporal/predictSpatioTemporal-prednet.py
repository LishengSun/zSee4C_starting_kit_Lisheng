#!/usr/bin/env python
# Usage: python predictSpatioTemporal.py step_num input_dir output_dir code_dir
# SEE4C CHALLENGE
#
# The input directory input_dir contains file X0.hdf, X1,hdf, etc. in HDF5 format.
# The output directory will receive the predicted values: Y0.hdf, Y1,hdf, etc.
# We expect people to predict the next 3 frames.
#
# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# The SEE4C CONSORTIUM, ITS ADVISORS, DATA DONORS AND CODE PROVIDERS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL THE SEE4C CONSORTIUM OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 
#
# Main contributors: Julio Jacques Jr., May 2017
# Use default location for the input and output data:
# If no arguments to this script is provided, this is where the data will be found
# and the results written to. This assumes the rood_dir is your local directory where
# this script is located in the starting kit.
import os
root_dir = os.getcwd()
default_input_dir = os.path.join(root_dir, "sample_data/")
default_output_dir = os.path.join(root_dir, "results/")
default_code_dir = root_dir
default_cache_dir = os.path.join(root_dir, "cache/")
#IMPORTANT: This determines whether we train or not (debug_mode 1 => no training, 0 => train the model) 
DEBUG_MODE=1
import time
import sys
def predictSpatioTemporal(step_num, input_dir, output_dir, code_dir, \
                          ext = '.h5', verbose=True, debug_mode=DEBUG_MODE, \
                          time_budget = 300, max_samples = 0, \
                          AR_order = 1, I_order = 0, MA_order = 0, \
                          num_predicted_frames=8, \
                          save_model = False, cache_data = False, \
                          cache_dir = "", \
                          version = "prednet" ):   
    ''' Main spatio-temporal prediction function.
    step_num
        Current file number n being processed Xn.h5.
    input_dir
        Input directory in which the training/adapatation data are found
        in two subdirectories train/ and adapt/
    output_dir
        Output directory in which we expect Yn+1.h5 predictions to be deposited.
        The next num_frame frames must be predicted.
    code_dir
        The directory to which the participant submissions are unzipped.
    ext
        The file extensions of input and output data
    verbose
        if True, debug messages are printed
    debug_mode
        0: run the code normally, train and test
        1: reload the model, do not train
    time_budget
        Maximum total running time in seconds. 
        The code should keep track of time spent and NOT exceed the time limit.
    max_samples
        Maximum number of training samples loaded. 
        Allows you to limit the number of traiining samples read for speed-up.
    Model order
        The order of an ARIMA model.
        Your training algorithm may be slow, so you may want to limit .
        the window of past frames used. 
        AR_order = 1 # Persistence is order 1
        I_order = 0
        MA_order = 0
    num_predicted_frames
        Number of frames to be predicted in the future.
    save_model 
        Models can eventually be pre-trained and re-loaded.
    cache_data
        Data that were loaded in the past can be cached in some 
        binary format for faster reload.
    cache_dir
        A directory where to cache data.
    version
        This code's version.
    '''
    #### Check whether everything went well (no time exceeded)
    execution_success = True
    start_time = time.time()         # <== Mark starting time
    #if not(cache_dir): cache_dir = os.path.join(code_dir, 'cache') # For the moment it is the code directory
    if not(cache_dir): 
        if os.path.isdir(os.path.join(code_dir, 'cache')):
            cache_dir = os.path.join(code_dir, 'cache') 
        else:
            cache_dir = code_dir
            
    sys.path.append (code_dir)
    sys.path.append (os.path.join(code_dir, 'utilities'))
    sys.path.append (os.path.join(code_dir, 'sample_code'))
    import data_io
    from data_io import vprint
   
    # Make a result directory and cache_dir if they do not exist
    data_io.mkdir(output_dir) 
    data_io.mkdir(cache_dir) 
    # List various directories
    if debug_mode >= 3:
        vprint( verbose,  "This code version is %d" + str(version))
        data_io.show_version()
        data_io.show_dir(os.getcwd()) # Run directory
        data_io.show_io(input_dir, output_dir)
        data_io.show_dir(output_dir) 
    
    # Our libraries  
    sys.path.append (code_dir)
           
    #### START WORKING ####  ####  ####  ####  ####  ####  ####  ####  ####  
    vprint( verbose,  "************************************************")
    vprint( verbose,  "******** Processing data chunk number " + str(step_num) + " ********")
    vprint( verbose,  "************************************************")
    ###########################################################################################    
    # KERAS CONFIGURATION FILE: Prednet initializations, added by Julio and modified by Isabelle
    if not os.path.exists(os.path.join(os.path.expanduser("~"), ".keras/")):
       print "warning: copying keras configuration file"
       data_io.mkdir(os.path.join(os.path.expanduser("~"), ".keras/"))
    data_io.cp(os.path.join(code_dir, "sample_code", "keras.json"), os.path.join("~", ".keras/"))
    ###########################################################################################
    from prednet_baseline import prednet_baseline as model
    # Hyper parameters are (we use cache_dir both as WEIGHTS_DIR and DATA_DIR):
    hyper_param = { 'cache_dir': cache_dir,
                    'version': 'faces_32_1',
                    'model_name': 'prednet_kitti',
                    'S_HELD': 10, # from 10 frames (hard-coded and FIXED in this current version)
                    'S_PRED': 1,  # predict the next 1 frame (hard-coded and FIXED in this current version)
                    'HORIZON': 11, # number of predicted files (predict 1 frame, then shift the data, and repeat): IMPORTANT: this variable was just tested with values = 8 and 11 (it does not work with values higher than 11, as S_HELD+S_PRED maximum value is fixed to 11 in this current version)
                    'TRAIN_SAMPLES': 0.95, # will use % of the train samples as train and the rest as validation
                    'test_batch_size': 10, 
                    'train_batch_size': 4,
                    'nb_epoch': 150, # default = 150
                    'samples_per_epoch': 500,
                    'nt': 10, 
                    'input_shape' : (1, 64, 64),
                    'input_vec_RTE' : 1916,
                    'stack_size' : (1, 48, 96, 192),
                    'A_filt_sizes' : (3, 3, 3),
                    'Ahat_filt_sizes' : (3, 3, 3, 3),
                    'R_filt_sizes' : (3, 3, 3, 3),
                    'layer_loss_weights' : [1., 0., 0., 0.],
                    'loss': 'mean_absolute_error',
                    'optimizer': 'adam',
                    'start_lr': 0.001,
                    'end_lr': 0.0001,
                    }
#    vprint( verbose,  "Building PREDNET predictions")
    M = model(hyper_param, code_dir, verbose=verbose)
 
    # ------------------------------------------------------------------- 
    # Read data and train the model
    ttrain_file = os.path.join(cache_dir, 'computed_time_train.txt')
    ttest_file = os.path.join(cache_dir, 'computed_time_test.txt')
    if step_num == 0:
        if debug_mode == 0:
#           vprint( verbose,  "Training PREDNET")
           start_time_train = time.time()
           output_file = open(ttrain_file, 'wb')
           M.fit(input_dir) 
           time_spent_train = time.time() - start_time_train
           output_file.write("computed time train: %s min"%float(time_spent_train/60))
           output_file.close()
    
    if step_num==0:
       output_file = open(ttest_file, 'wb')
    else:
       output_file = open(ttest_file, 'r')
    
    # predict 
    start_time_test = time.time()
#    vprint( verbose,  "Running PREDNET predictions")
    M.predict(input_dir,output_dir,step_num)
    time_spent_test = time.time() - start_time_test
    if step_num==0:
       output_file.write("%s"%float(time_spent_test/60))
       output_file.close()
    else:
       acc_test_time = float(output_file.read()) + time_spent_test/60
       output_file.close()
       os.system('rm ' + ttest_file)
       output_file = open(ttest_file, 'wb')
       output_file.write("%s"%float(acc_test_time))
       output_file.close()
    # ---------------------------------------------------------------------
                    
    time_spent = time.time() - start_time
    time_left_over = time_budget - time_spent
    if time_left_over>0:
        vprint( verbose,  "[+] Done")
        vprint( verbose,  "[+] Time spent %5.2f sec " % time_spent + "::  Time budget %5.2f sec" % time_budget)
    else:
        execution_success = 0
        vprint( verbose,  "[-] Time exceeded")
        vprint( verbose,  "[-] Time spent %5.2f sec " % time_spent + " > Time budget %5.2f sec" % time_budget)
              
    return execution_success
    
# =========================== BEGIN PROGRAM ================================
if __name__=="__main__":    
    if len(sys.argv)==1: # Use the default if no arguments are provided
        step_num = 0
        input_dir = default_input_dir
        output_dir = default_output_dir
        code_dir = default_code_dir
        cache_dir= default_cache_dir
        debug_mode = DEBUG_MODE
        running_locally = True
    else:
        step_num = int(sys.argv[1])
        input_dir = sys.argv[2]
        output_dir = os.path.abspath(sys.argv[3])
        code_dir = sys.argv[4]
        if len(sys.argv)<6:
            debug_mode = DEBUG_MODE
        else:
            debug_mode = sys.argv[5]
        cache_dir=""
        running_locally = False
        
    execution_success = predictSpatioTemporal(step_num, input_dir, output_dir, code_dir, \
                                              cache_dir=cache_dir, debug_mode=debug_mode)
    if not running_locally: 
        if execution_success:
            exit(0)
        else:
            exit(1)#!/usr/bin/env python
# Usage: python predictSpatioTemporal.py step_num input_dir output_dir code_dir
# SEE4C CHALLENGE
#
# The input directory input_dir contains file X0.hdf, X1,hdf, etc. in HDF5 format.
# The output directory will receive the predicted values: Y0.hdf, Y1,hdf, etc.
# We expect people to predict the next 3 frames.
#
# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# The SEE4C CONSORTIUM, ITS ADVISORS, DATA DONORS AND CODE PROVIDERS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL THE SEE4C CONSORTIUM OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 
#
# Main contributors: Julio Jacques Jr., May 2017
# Use default location for the input and output data:
# If no arguments to this script is provided, this is where the data will be found
# and the results written to. This assumes the rood_dir is your local directory where
# this script is located in the starting kit.
import os
root_dir = os.getcwd()
default_input_dir = os.path.join(root_dir, "sample_data/")
default_output_dir = os.path.join(root_dir, "results/")
default_code_dir = root_dir
default_cache_dir = os.path.join(root_dir, "cache/")
#IMPORTANT: This determines whether we train or not (debug_mode 1 => no training, 0 => train the model) 
DEBUG_MODE=1
import time
import sys
def predictSpatioTemporal(step_num, input_dir, output_dir, code_dir, \
                          ext = '.h5', verbose=True, debug_mode=DEBUG_MODE, \
                          time_budget = 300, max_samples = 0, \
                          AR_order = 1, I_order = 0, MA_order = 0, \
                          num_predicted_frames=8, \
                          save_model = False, cache_data = False, \
                          cache_dir = "", \
                          version = "prednet" ):   
    ''' Main spatio-temporal prediction function.
    step_num
        Current file number n being processed Xn.h5.
    input_dir
        Input directory in which the training/adapatation data are found
        in two subdirectories train/ and adapt/
    output_dir
        Output directory in which we expect Yn+1.h5 predictions to be deposited.
        The next num_frame frames must be predicted.
    code_dir
        The directory to which the participant submissions are unzipped.
    ext
        The file extensions of input and output data
    verbose
        if True, debug messages are printed
    debug_mode
        0: run the code normally, train and test
        1: reload the model, do not train
    time_budget
        Maximum total running time in seconds. 
        The code should keep track of time spent and NOT exceed the time limit.
    max_samples
        Maximum number of training samples loaded. 
        Allows you to limit the number of traiining samples read for speed-up.
    Model order
        The order of an ARIMA model.
        Your training algorithm may be slow, so you may want to limit .
        the window of past frames used. 
        AR_order = 1 # Persistence is order 1
        I_order = 0
        MA_order = 0
    num_predicted_frames
        Number of frames to be predicted in the future.
    save_model 
        Models can eventually be pre-trained and re-loaded.
    cache_data
        Data that were loaded in the past can be cached in some 
        binary format for faster reload.
    cache_dir
        A directory where to cache data.
    version
        This code's version.
    '''
    #### Check whether everything went well (no time exceeded)
    execution_success = True
    start_time = time.time()         # <== Mark starting time
    #if not(cache_dir): cache_dir = os.path.join(code_dir, 'cache') # For the moment it is the code directory
    if not(cache_dir): 
        if os.path.isdir(os.path.join(code_dir, 'cache')):
            cache_dir = os.path.join(code_dir, 'cache') 
        else:
            cache_dir = code_dir
            
    sys.path.append (code_dir)
    sys.path.append (os.path.join(code_dir, 'utilities'))
    sys.path.append (os.path.join(code_dir, 'sample_code'))
    import data_io
    from data_io import vprint
   
    # Make a result directory and cache_dir if they do not exist
    data_io.mkdir(output_dir) 
    data_io.mkdir(cache_dir) 
    # List various directories
    if debug_mode >= 3:
        vprint( verbose,  "This code version is %d" + str(version))
        data_io.show_version()
        data_io.show_dir(os.getcwd()) # Run directory
        data_io.show_io(input_dir, output_dir)
        data_io.show_dir(output_dir) 
    
    # Our libraries  
    sys.path.append (code_dir)
           
    #### START WORKING ####  ####  ####  ####  ####  ####  ####  ####  ####  
    vprint( verbose,  "************************************************")
    vprint( verbose,  "******** Processing data chunk number " + str(step_num) + " ********")
    vprint( verbose,  "************************************************")
    ###########################################################################################    
    # KERAS CONFIGURATION FILE: Prednet initializations, added by Julio and modified by Isabelle
    if not os.path.exists(os.path.join(os.path.expanduser("~"), ".keras/")):
       print "warning: copying keras configuration file"
       data_io.mkdir(os.path.join(os.path.expanduser("~"), ".keras/"))
    data_io.cp(os.path.join(code_dir, "sample_code", "keras.json"), os.path.join("~", ".keras/"))
    ###########################################################################################
    from prednet_baseline import prednet_baseline as model
    # Hyper parameters are (we use cache_dir both as WEIGHTS_DIR and DATA_DIR):
    hyper_param = { 'cache_dir': cache_dir,
                    'version': 'faces_32_1',
                    'model_name': 'prednet_kitti',
                    'S_HELD': 10, # from 10 frames (hard-coded and FIXED in this current version)
                    'S_PRED': 1,  # predict the next 1 frame (hard-coded and FIXED in this current version)
                    'HORIZON': 11, # number of predicted files (predict 1 frame, then shift the data, and repeat): IMPORTANT: this variable was just tested with values = 8 and 11 (it does not work with values higher than 11, as S_HELD+S_PRED maximum value is fixed to 11 in this current version)
                    'TRAIN_SAMPLES': 0.95, # will use % of the train samples as train and the rest as validation
                    'test_batch_size': 10, 
                    'train_batch_size': 4,
                    'nb_epoch': 150, # default = 150
                    'samples_per_epoch': 500,
                    'nt': 10, 
                    'input_shape' : (1, 64, 64),
                    'input_vec_RTE' : 1916,
                    'stack_size' : (1, 48, 96, 192),
                    'A_filt_sizes' : (3, 3, 3),
                    'Ahat_filt_sizes' : (3, 3, 3, 3),
                    'R_filt_sizes' : (3, 3, 3, 3),
                    'layer_loss_weights' : [1., 0., 0., 0.],
                    'loss': 'mean_absolute_error',
                    'optimizer': 'adam',
                    'start_lr': 0.001,
                    'end_lr': 0.0001,
                    }
#    vprint( verbose,  "Building PREDNET predictions")
    M = model(hyper_param, code_dir, verbose=verbose)
 
    # ------------------------------------------------------------------- 
    # Read data and train the model
    ttrain_file = os.path.join(cache_dir, 'computed_time_train.txt')
    ttest_file = os.path.join(cache_dir, 'computed_time_test.txt')
    if step_num == 0:
        if debug_mode == 0:
#           vprint( verbose,  "Training PREDNET")
           start_time_train = time.time()
           output_file = open(ttrain_file, 'wb')
           M.fit(input_dir) 
           time_spent_train = time.time() - start_time_train
           output_file.write("computed time train: %s min"%float(time_spent_train/60))
           output_file.close()
    
    if step_num==0:
       output_file = open(ttest_file, 'wb')
    else:
       output_file = open(ttest_file, 'r')
    
    # predict 
    start_time_test = time.time()
#    vprint( verbose,  "Running PREDNET predictions")
    M.predict(input_dir,output_dir,step_num)
    time_spent_test = time.time() - start_time_test
    if step_num==0:
       output_file.write("%s"%float(time_spent_test/60))
       output_file.close()
    else:
       acc_test_time = float(output_file.read()) + time_spent_test/60
       output_file.close()
       os.system('rm ' + ttest_file)
       output_file = open(ttest_file, 'wb')
       output_file.write("%s"%float(acc_test_time))
       output_file.close()
    # ---------------------------------------------------------------------
                    
    time_spent = time.time() - start_time
    time_left_over = time_budget - time_spent
    if time_left_over>0:
        vprint( verbose,  "[+] Done")
        vprint( verbose,  "[+] Time spent %5.2f sec " % time_spent + "::  Time budget %5.2f sec" % time_budget)
    else:
        execution_success = 0
        vprint( verbose,  "[-] Time exceeded")
        vprint( verbose,  "[-] Time spent %5.2f sec " % time_spent + " > Time budget %5.2f sec" % time_budget)
              
    return execution_success
    
# =========================== BEGIN PROGRAM ================================
if __name__=="__main__":    
    if len(sys.argv)==1: # Use the default if no arguments are provided
        step_num = 0
        input_dir = default_input_dir
        output_dir = default_output_dir
        code_dir = default_code_dir
        cache_dir= default_cache_dir
        debug_mode = DEBUG_MODE
        running_locally = True
    else:
        step_num = int(sys.argv[1])
        input_dir = sys.argv[2]
        output_dir = os.path.abspath(sys.argv[3])
        code_dir = sys.argv[4]
        if len(sys.argv)<6:
            debug_mode = DEBUG_MODE
        else:
            debug_mode = sys.argv[5]
        cache_dir=""
        running_locally = False
        
    execution_success = predictSpatioTemporal(step_num, input_dir, output_dir, code_dir, \
                                              cache_dir=cache_dir, debug_mode=debug_mode)
    if not running_locally: 
        if execution_success:
            exit(0)
        else:
            exit(1)