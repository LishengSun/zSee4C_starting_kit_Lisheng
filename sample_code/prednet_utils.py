import numpy as np
from keras import backend as K
from keras.preprocessing.image import Iterator


# Data generator that creates sequences for input into PredNet.
class SequenceGenerator(Iterator):
    def __init__(self, data_file, source_file, nt,
                 batch_size=8, shuffle=False, seed=None,
                 output_mode='error', sequence_start_mode='all', N_seq=None,
                 dim_ordering=K.image_dim_ordering()):
        self.X = hkl.load(data_file)  # X will be like (n_images, nb_cols, nb_rows, nb_channels)
        self.sources = hkl.load(source_file) # source for each image so when creating sequences can assure that consecutive frames are from same video
        self.nt = nt
        self.batch_size = batch_size
        self.dim_ordering = dim_ordering
        assert sequence_start_mode in {'all', 'unique'}, 'sequence_start_mode must be in {all, unique}'
        self.sequence_start_mode = sequence_start_mode
        assert output_mode in {'error', 'prediction'}, 'output_mode must be in {error, prediction}'
        self.output_mode = output_mode

        if self.dim_ordering == 'th':
            self.X = np.transpose(self.X, (0, 3, 1, 2))
        self.im_shape = self.X[0].shape

        if self.sequence_start_mode == 'all':  # allow for any possible sequence, starting from any frame
            self.possible_starts = np.array([i for i in range(self.X.shape[0] - self.nt) if self.sources[i] == self.sources[i + self.nt - 1]])
        elif self.sequence_start_mode == 'unique':  #create sequences where each unique frame is in at most one sequence
            curr_location = 0
            possible_starts = []
            while curr_location < self.X.shape[0] - self.nt + 1:
                if self.sources[curr_location] == self.sources[curr_location + self.nt - 1]:
                    possible_starts.append(curr_location)
                    curr_location += self.nt
                else:
                    curr_location += 1
            self.possible_starts = possible_starts

        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)
        if N_seq is not None and len(self.possible_starts) > N_seq:  # select a subset of sequences if want to
            self.possible_starts = self.possible_starts[:N_seq]
        self.N_sequences = len(self.possible_starts)
        super(SequenceGenerator, self).__init__(len(self.possible_starts), batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        batch_x = np.zeros((current_batch_size, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(index_array):
            idx = self.possible_starts[idx]
            batch_x[i] = self.preprocess(self.X[idx:idx+self.nt])
        if self.output_mode == 'error':  # model outputs errors, so y should be zeros
            batch_y = np.zeros(current_batch_size, np.float32)
        elif self.output_mode == 'prediction':  # output actual pixels
            batch_y = batch_x
        return batch_x, batch_y

    def preprocess(self, X):
        return X.astype(np.float32) / 255

    def create_all(self):
        X_all = np.zeros((self.N_sequences, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(self.possible_starts):
            X_all[i] = self.preprocess(self.X[idx:idx+self.nt])
        return X_all


# Data generator that creates sequences for input into PredNet.
class SequenceGeneratorChalearn(Iterator):
    def __init__(self, data_file, nt, unique_start=False, batch_size=8, shuffle=False, seed=None, output_mode='error', N_seq=None, dim_ordering=K.image_dim_ordering()):
        assert output_mode in {'error', 'prediction'}, 'output_mode must be in {error, prediction}'

        # Load data, check number of sequences and sequence length
        self.X = np.transpose(np.load(data_file), (0, 1, 3, 4, 2))  # X will be like (n_seq, n_frames, nb_cols, nb_rows, nb_channels)
        s_num, s_len = self.X.shape[0], self.X.shape[1]

        # Store general sampler parameters
        self.nt = nt  # Sample length
        self.batch_size = batch_size
        self.dim_ordering = dim_ordering
        self.output_mode = output_mode

        # Shuffle data dimensions to match ordering expected by the framework
        self.X = np.transpose(self.X, (0, 1, 4, 2, 3)) if self.dim_ordering == 'th' else self.X
        self.im_shape = self.X.shape[-3:]

        # Prepare possible sequence starts
        self.possible_starts = np.arange(0, s_num*s_len, nt) if unique_start else np.where(
            np.concatenate((np.ones((s_num, s_len-nt)), np.zeros((s_num, nt))), axis=1).flatten() == 1
        )[0]
        self.possible_starts = np.random.permutation(self.possible_starts) if shuffle else self.possible_starts
        self.possible_starts = self.possible_starts if N_seq is None else self.possible_starts[:N_seq]
        self.N_sequences = len(self.possible_starts)

        # Merge into single sequence for fast sampling, initialize super-class
        self.X = np.reshape(self.X, (s_num*s_len,)+self.im_shape)
        super(SequenceGeneratorChalearn, self).__init__(self.N_sequences, batch_size, shuffle, seed)

    def next(self):
        # Retrieve next sample indices
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        # Sample data given the indices
        batch_x = np.zeros((current_batch_size, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(index_array):
            idx = self.possible_starts[idx]
            batch_x[i] = self.preprocess(self.X[idx:idx+self.nt])

        # Prepare targets and return
        batch_y = np.zeros(current_batch_size, np.float32) if self.output_mode == 'error' else batch_x
        return batch_x, batch_y

    def preprocess(self, X):
        # Normalize data and transform to float32
        #return X.astype(np.float32) / 255   # original (when input data are in the range [0,...,255])
        return X.astype(np.float32)  # edited by Julio (when input data are in the range [0,...,1])

    def create_all(self):
        X_all = np.zeros((self.N_sequences, self.nt) + self.im_shape, dtype=np.float32)
        for i, idx in enumerate(self.possible_starts):
            X_all[i] = self.preprocess(self.X[idx:idx+self.nt])
        return X_all
