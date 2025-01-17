# Functions performing various input/output operations for the See.4C challenge
# Main contributors: Stephane Ayache, Isabelle Guyon and Lisheng Sun

# Date: January 2017


# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, SEE.4C, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 



import numpy as np # We will use numoy arrays
try:
	import cPickle as pickle
except:
	import pickle
import os # use os.path.join to concatenate strings in directory names
import time
import h5py
from data_io import vprint
import sys
import traceback


class DataManager:
    
    ''' This class aims at loading, saving, and displaying data.
    
    Data members:
    datatype = one of "public", "feedback", or "validate"
    X = data matrix, samples in lines (time index increasing with line), features in columns.
    t = time index. The time index may show ruptures e.g. 0, 1, 2, 3, 0, 1, 2, 3; indicating cuts.
     
    Methods defined:
    __init__ (...)
        x.__init__([(feature, value)]) -> void		
        Initialize the data members with the tuples (feature, value) given as argument. An unlimited number of tuples can be passed as argument.
        If input_dir is given, calls loadTrainData.
        Parameters:
        two_d_map: bool. Maps 1d array data to 2d using a pre-trained SOM (Self Organizing Map) map when True (default).
        map_file: the SOM map used when two_d_map. default ='./utilities/Midx_199_44by44.txt'
        
    loadTrainData (...)
        x.loadData (input_dir, max_samples=float('inf'), verbose="True") -> success		
        Load all the training samples found in directory input_dir/train. 
        Ignores the samples in input_dir/adapt, if any.
        input_dir/train may contain multiple subdirectories, Xmn/.
        The data must be read from all of them, it order of DECREASING values of n.
        The directories contains files Xn.h5, which must be read in order if INCREASING n values.
        If data are already loaded, this function overwrites X, unless append="True".
        For speed reasons, stops after max_samples samples (frames) have been read.
        Returns success="True/False".
        =============================
        loadTrainData() returns a ndarray X and a time index array t
            X.shape(total_num_of_frames=101*125frames/videos, 32,32)
            t = array([0, ...., 124, 0, ...., 124,...]), t.shape=(total_num_of_frames,)
		
    appendSamples (...)
        x.appendSamples (chunk_num, input_dir, verbose="True") -> success		
        Append to X all the samples found in directory input_dir/adapt/Xn.h5, where n=chunk_num.
        Returns success="True/False".        
        
    getInfo (...)
        x.getInfo () -> string	
        Pretty prints information about the object.

    saveData() 
        save read data (array X, T) to pickle or h5 file		
    '''
	
    def __init__(self, datatype="unknown", data_file="", verbose=False, max_samples=float('inf'), cache_file="", two_d_map=True, map_file='./utilities/Midx_199_44by44.txt'):
        '''Constructor'''
        self.version = "1"
        self.datatype = datatype 
        self.verbose = verbose
        self.max_samples=max_samples
        self.two_d_map = two_d_map
        self.map_file = map_file
        self.cache_file=cache_file # To save/reload data in binary format (only if not empty)
        if not cache_file: 
            self.use_pickle = False
        else:
            self.use_pickle = True
        self.X = np.array([])
        self.t = np.array([])
        vprint(self.verbose, "Data Manager :: Version = " + self.version)
        if data_file:
            self.loadData(data_file)

           
    def __repr__(self):
        return "DataManager :\n\t" + str(self.X.__repr__) + "\n\t" + str(self.t.__repr__)

    def __str__(self):
        val = "DataManager :\n" + self.getInfo
        return val
  
    def getInfo(self):
        '''A nice string with information about the data.'''       
        val = ""
        return val
        
    def loadTrainData(self, data_dir="", max_samples=float('inf')):
        ''' Get the data from hdf5 files.'''
        success = True
        data_reloaded = False
        vprint(self.verbose, "Data Manager :: ========= Reading training data from " + data_dir)
        start = time.time()
        vid=0
        if self.use_pickle and self.reloadData(self.cache_file):
            # Try to reload the file from a pickle
            data_reloaded = True # Turn "success" to false if there is a problem.
        else:
            # Load the data into X and t.
            dir_list = []
            for dir in os.listdir(data_dir):
                if os.path.isdir(os.path.join(data_dir, dir)):
                    dir_list.append(dir)
            # sort dir in decreasing order of n for n in Xmn        
            dir_list = sorted(dir_list, key=lambda i: i.split('m')[-1], reverse=True)
            vprint(self.verbose, dir_list)
            self.X=np.array([]) # Re-initialize from scratch
            self.t=np.array([])
            for dir in dir_list:
                for data_file in sorted([h5file for h5file in os.listdir(os.path.join(data_dir, dir)) if h5file.endswith('h5')],key=lambda i:int(i.split('.')[0].split('X')[-1])):
                    vprint(self.verbose, "Loading %s"%data_file)
                    self.appendSamples(data_file, os.path.join(data_dir, dir), verbose=False)
                    vid=vid+1
                   
        if self.use_pickle and not data_reloaded:
            # Save data as a pickle for "faster" later reload
            self.saveData(self.cache_file, format='pickle')
            
        end = time.time()
        if len(self.X)==0:
            success = False 
            vprint(self.verbose, "[-] Loading failed")
        else:
            vprint(self.verbose, "[+] Success, loaded %d videos in %5.2f sec" % (vid, end - start))
        return success
	

    def appendSamples(self, data_file, data_dir="", verbose=False):
        ''' After loading training samples, get additional data from the adapt directory.
        data_file: Number n of the 'chunk' or 'step' (appearing in the file name)
        Alternatively, the full file name Xn can be supplied as a string instead of the chunk number.
        '''
        success = True
        start = time.time()
        # Append new data to X and t.
        if isinstance(data_file, int):
            data_file = "X" + str(data_file)
        vprint(self.verbose and verbose, "Data Manager :: ========= Appending samples " + data_dir +  data_file)
        X_add, t_add = self.getOneSample(data_file, data_dir)
        #if t_add[0]==0: print "New sequence"
        #print "%s %d %d %d" % (data_file, np.min(t_add), np.max(t_add), len(t_add))
            
        if len(self.X)==0:
            self.X = X_add
            self.t = t_add
        else:
            self.X = np.vstack((self.X, X_add))

            if t_add[0][0] == 0:
                t_add = np.array([t+1+self.t[-1][0] for t in t_add]).reshape(t_add.shape)
            self.t = np.vstack((self.t, t_add))

        end = time.time()
        if len(self.X)==0:
            success = False 
            vprint(self.verbose and verbose, "[-] Loading failed")
        else:
            vprint(self.verbose and verbose, "[+] Success in %5.2f sec" % (end - start))
        return success
        
    def loadData(self, data_file, data_dir=""):
        ''' Erase previous data and load data from a give data file.
        data_file: Number n of the 'chunk' or 'step' (appearing in the file name)
        Alternatively, the full file name Xn can be supplied as a string instead of the chunk number.
        '''
        success = True
        start = time.time() 
        if isinstance(data_file, int):
            data_file = "X" + str(data_file)
        vprint(self.verbose, "Data Manager :: ========= Loading data from " + data_file)          
        self.X, self.t = self.getOneSample(data_file, data_dir)
        end = time.time()
        if len(self.X)==0:
            success = False 
            vprint(self.verbose, "[-] Loading failed")
        else:
            vprint(self.verbose, "[+] Success in %5.2f sec" % (end - start))
        
        return success
        
    def getOneSample(self, data_file, data_dir=""):
        ''' Return one video read from hdf5 format: 
        Parameters: 
            data_file: file name (no extention)
            data_dir: data path
        '''
        try:
            # try:
            #     f_0 = h5py.File('./sample_data/train/Xm1/X1.h5','r')#This is the first file and will give the first timestamp t0
            #     print data_dir
            #     print os.path.join(data_dir,'train/Xm1/X1.h5')
            #     # f_0 = h5py.File(os.path.join(data_dir,'train/Xm1/X1.h5'),'r')
            #     t_0 = np.array(f_0['X']['value']['t']['value'][:])[0][0]
            # except: # use the first file in /train/Xm1 as starting file
            #     f_0 = sorted(os.listdir('./sample_data/train/Xm1/'), key=lambda f:int(f.split('.')[0].split('X')[-1]))[0]
            #     t_0 = np.array(f_0['X']['value']['t']['value'][:])[0][0]
            #     vprint (self.verbose, "Cannot load train/Xm1/X1.h5 to initialize time indexing. Instead use the first file in train/Xm1/.")
            
            t_0 = 1246492800.0
            
            if not data_file.endswith('.h5'):
                data_file = data_file + '.h5'
            f = h5py.File(os.path.join(data_dir, data_file),'r')
            try:
                X = np.array(f['X']['value']['X']['value'][:])
                
                ### corrected by Lisheng ####
                # if 't' in f: 
                #     X = np.array(f['t']['value']['t']['value'][:])
                if 't' in f['X']['value']:
                    t_absolute = np.array(f['X']['value']['t']['value'][:])
                    # print t_absolute[:,0].tolist()
                    # print t_0
                    # t = [int(float(t_abs-t0)/300) for t_abs in t_absolute.tolist()]
                    t = [(t_abs-t_0)/300 for t_abs in t_absolute[:,0].tolist()]
                    # print t
                    t = np.array(t).reshape(t_absolute.shape)

                ### corrected by Lisheng ####
                else:
                    t = np.array(range(X.shape[0]))
                
            except: # Lisheng's simpler format
                try:
                    X = np.array(f['X'][:])
                    if 't' in f: 
                        t = np.array(f['t'][:])   
                    else:
                        t = np.array(range(X.shape[0])) 
                except Exception:
                    vprint(self.verbose, traceback.format_exc())
                    X = np.array([])
                    t = np.array([])  
                
            if len(t)==0 or len(t)!=len(X):
                t = np.array(range(X.shape[0])) 
            if len(X.shape) > 3: # turn to gray levels
                X = X[:,:,:,0]
            if self.two_d_map:
                # X_1d = np.hstack((X,t))
                # print X_1d.shape
                # vprint (self.verbose, 'Mapping X to 2D')
                X = self.two_d_mapping(X)
        except Exception:
            print(traceback.format_exc())


        return (X, t)
        
    def saveData(self, data_file, data_dir="", frames=[], format='pickle', map_to_1d=True, map_to_1d_when_starts_with='Y'):
        ''' Save data in picke / h5 format.        Parameters: 
            data_file: save data under this filename (no extention)
            data_dir: where to save data
            frames: specify which lines in the video matrix to be saved,  
            e.g. frames=(start_frame, end_frame)=(10,15)
                    default = entire video matrix
            format: 'pickle' or 'h5', default = 'pickle'
            map_to_1d: bool. True (by default): map the predictions back to 1d when they are saved with a filename starting with the value of parameter 'map_to_1d_when_starts_with'; False: save data as it is.
            map_to_1d_when_starts_with: str. Indicates the starting part of filename which will get mapped to 1d when map_to_1d=True
        '''
        if not data_file.endswith(format):
            data_file = data_file + '.' + format
        success = True
        if data_file.startswith(map_to_1d_when_starts_with) and map_to_1d:
            two_d_map = False
            self.X = self.map_back_to_1d(self.X)
        try:
            filename = os.path.join(data_dir, data_file)
            vprint(self.verbose, "Data Manager :: ========= Saving data to " + filename)

            start = time.time()
            # Write some code to save the data
            if frames: 
                if format=='h5': 
                    with h5py.File(filename, 'w') as f:
                        f.create_dataset(name='X', shape=self.X[frames[0]:frames[1]].shape, \
                            data=self.X[frames[0]:frames[1]])
                        f.create_dataset(name='t', shape=self.t[frames[0]:frames[1]].shape, \
                            data=self.t[frames[0]:frames[1]])
                else: 
                    with open(filename, 'wb') as f:
                        dict_to_save = {key:self.__dict__[key] for key in self.__dict__.keys() if not key in ['X', 't']}
                        dict_to_save['X'] = self.X[frames[0]:frames[1]]
                        dict_to_save['t'] = self.t[frames[0]:frames[1]]
                        pickle.dump(dict_to_save, f, 2)
            else: #save the entire matrix
                if format=='h5':
                    with h5py.File(filename, 'w') as f:
                        f.create_dataset(name='X', shape=self.__dict__['X'].shape, data=self.__dict__['X'])
                        f.create_dataset(name='t', shape=self.__dict__['t'].shape, data=self.__dict__['t'])
                else: 
                    with open(filename, 'wb') as f:
                        pickle.dump(self.__dict__, f, 2) 
        except Exception as e: 
            vprint (self.verbose, e)
            success = False
        end = time.time()
        vprint(self.verbose, "[+] Success in %5.2f sec" % (end - start))
        return success


    def reloadData(self, filename="", data_dir=""):
        ''' Reload data in pickle format.'''
        success = True
        vprint(self.verbose, "Data Manager :: ========= Reloading data from " + filename)
        start = time.time()
        # Write some code to reload the data
        temp =[]
        try:
            if filename.endswith('h5'): 
                with h5py.File(os.path.join(data_dir, filename), 'r') as f:
                    self.X = f['X'][:]
                    self.t = f['t'][:]
            elif filename.endswith('pickle'):
                with open(os.path.join(data_dir, filename), 'rb') as f:
                    temp = pickle.load(f)
                    self.X = temp['X']
                    self.t = temp['t']
                    vprint(self.verbose, filename)
            else:
                success = False
                vprint(self.verbose, "[-] No such file extension." + filename)            
        except Exception as e: 
            vprint (self.verbose, e)
            success = False 
        end = time.time()
        if success:
            vprint(self.verbose, "[+] Success in %5.2f sec" % (end - start))
        return success

    def browse(self):
        ''' Open a data browser to browse through the data.'''        

    def play_video(self):
        '''play video in python:
        http://stackoverflow.com/questions/21313542/how-to-open-a-mp4-file-with-python
        '''

    def motion_history_image(self, start=0, end=10, step=1, tau=10, d=0):
        """
        start and end and the first and last frame. step is the stride.
        tau = 50 means we consider only motions taken within 50 frames.
        d = difference threshold: 
        if the difference between 2 images < d at a point (x,y), it's considered a motionless point at t
        """
        import matplotlib.pyplot as plt
        # get video frames 
        nmax = len(self.X)
        if end>nmax: end=nmax
        if end<start: end=start
        frame_index_to_display = range(start, end+1, step)
        frame_list = self.X[frame_index_to_display]
        # compute difference images
        difference_images = np.asarray([frame_list[t+1] - frame_list[t] for t in range(len(frame_list)-1)])
        tmax,xmax,ymax = difference_images.shape
        # initialize the motion history image
        MHI = np.zeros((xmax, ymax))
        #loop over time
        for t in range(tmax):        
            #loop each position
            for x in range(xmax):
                for y in range(ymax):
                    if difference_images[t,x,y] > d: # if moving now
                        # pixel value = max value = tau
                        MHI[x,y] = tau 
                    else: # if motionless now
                        # pixel value decays by 1
                        MHI[x,y] = max(0, MHI[x,y]-1)         
        plt.imshow(MHI)
        plt.colorbar()
        plt.show()
        return MHI
        

    def display(self, start=0, end=0, step=1):
        ''' Display frames graphically in a nice way.
            start and end and the first and last frame. step is the stride.
            self.X is a list of array, each array with shape (32, 32).'''
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plot_i=1
        nmax = len(self.X)
        if end>nmax: end=nmax
        if end<start: end=start
        frame_index_to_display = range(start, end+1, step)
        fnum = len(frame_index_to_display)
        for i in range(fnum):
            sf = fig.add_subplot(1, fnum, i+1)
            sf.imshow(self.X[frame_index_to_display[i]], cmap='gray', interpolation='None')
            plot_i += 1
            sf.axis('off')
            sf.set_title(str(frame_index_to_display[i]))
        plt.show()


    def two_d_mapping(self, X_1d):
        """
        map 1-d array to a 2-d space with Self Organizing Maps technique. The map used here is defined by self.map_file
        """
        try:
            Midx = np.loadtxt(self.map_file)
            T = X_1d.shape[0]
            n = Midx.shape[0] #or[1], Midx is square
            X_2d = np.empty((T,n,n))
            for i in range(n):
                for j in range(n):
                    X_2d[:,i,j] = X_1d[:, int(Midx[i,j])].reshape((T,))
            return X_2d

        except Exception:
            vprint(self.verbose, traceback.format_exc())
            vprint (self.verbose, "Error \n Data has not been transformed to 2D array.") 
            return X_1d
             # retrain the map

    def map_back_to_1d(self, X_2d):
        """
        map 2-d array back to 1-d using the same maps for 1-d to 2-d (defined by self.map_file). This is useful when saving predictions
        for the reason of scoring.

        """
        try:
            Midx = np.loadtxt(self.map_file)
            try:
                original_Xshape = np.array(h5py.File('./sample_data/train/Xm1/X1.h5','r')['X']['value']['X']['value'][:]).shape
            except:
                vprint (self.verbose, "Error: Cannot load ./sample_data/train/Xm1/X1.h5.\n Use the default value (1916, 1) for 1d array") 
                original_Xshape = (12, 1916, 1)
            # print original_Xshape
            # print X_2d.shape
            T, n, _ = X_2d.shape
            X_1d = np.empty((T, original_Xshape[1], original_Xshape[2]))
            T,m,_ = X_1d.shape
            
            where = 0
            while where < m:
                for i in range(n):
                    for j in range(n):
                        if int(Midx[i,j]) == where:
                            X_1d[:, where]=X_2d[:,i,j].reshape(X_1d[:, where].shape)
                where += 1
            return X_1d
            

        except Exception:
            vprint(self.verbose, traceback.format_exc())
            vprint (self.verbose, "Error \n Data has not been transformed back to 1D array.") 
            return X_2d









