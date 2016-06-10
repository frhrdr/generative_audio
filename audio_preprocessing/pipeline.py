from __future__ import print_function
from cconfig import config
from numpy.fft import fft
import numpy as np
import os
import scipy.io.wavfile as wav
import scipy.signal as sg
import glob
import matplotlib.pyplot as plt


def load_matrix(folder_spec, data):

    numpy_file = config.datapath + folder_spec + data + '.npy'
    with open(numpy_file, 'rb') as fs:
        np_data = np.load(fs)

        for obj_id in np_data:
            print(obj_id)
            if obj_id == 'x_data':
                data1 = np_data[obj_id]
            elif obj_id == 'y_data':
                data2 = np_data[obj_id]
            elif obj_id == 'mean_x':
                data1 = np_data[obj_id]
            elif obj_id == 'std_x':
                data2 = np_data[obj_id]

    return data1, data2


def convert_nd_audio_to_sample_blocks(nd_audio, block_size):
    block_lists = []
    total_samples = nd_audio.shape[0]
    num_samples_so_far = 0
    while num_samples_so_far < total_samples:
        block = nd_audio[num_samples_so_far:num_samples_so_far + block_size]
        if block.shape[0] < block_size:
            padding = np.zeros((block_size - block.shape[0],))
            block = np.concatenate((block, padding))
        block_lists.append(block)
        num_samples_so_far += block_size
    return block_lists


class AudioPipeline(object):

    def __init__(self, folder_spec='', n_to_load=1, highest_freq=440, clip_len=2, mat_dirs=None, chunks_per_sec=4,
                 down_sampling=False):
        self._down_sampling = down_sampling
        self.raw_audios = []
        self.num_of_files = 0
        self._sampled_audios = []
        self._root_path = config.datapath + folder_spec
        self.def_highest_freq = highest_freq
        self._offset = 2
        self._n_to_load = n_to_load
        self._folder_spec = folder_spec
        self._clip_length = clip_len # all audio's will be cut to the same clip length (in seconds)
        self.new_sample_rate = 0 # to be determined during down sampling method
        self.block_size = 0      # to be determined later, depends on new sample rate/frequency
        self.load_data()
        if self._down_sampling:
            self.down_sampling()
        else:
            self.new_sample_rate = config.frequency_of_format

        self.chunks_per_sec = chunks_per_sec
        self._train_signal_pairs = None
        self.signal_mean_std = None
        self._train_spectra_pairs = None

        if mat_dirs is not None:
            try:
                self._train_signal_pairs = load_matrix(mat_dirs[0], mat_dirs[1])
                if len(mat_dirs) is 3:
                   self._train_spectra_pairs = load_matrix(mat_dirs[0], mat_dirs[2])
            except IOError as e:
                print("could not read matrix or spectra files. so they were not created")

    def load_data(self):

        print("Loading audio files from: %s" % self._root_path)
        os.chdir(self._root_path)
        files_to_load = glob.glob("*.wav")
        # print("# files to load %d (max to load %d)" % (len(files_to_load), self._n_to_load ))
        if len(files_to_load) > self._n_to_load:
            files_to_load = files_to_load[:self._n_to_load]
        else:
            self._n_to_load = len(files_to_load)

        for audio in files_to_load:
            audio_file = os.path.join(self._root_path, audio)

            try:
                # read file
                sample_rate, nd_audio = wav.read(audio_file)
                # some of the samples we use are recorded in stereo, but the signal is actually mono
                # so we can just skip one channel
                if nd_audio.ndim > 1:
                    if nd_audio.shape[1] == 2:
                        nd_audio = nd_audio[:, 0]
                audio = AudioSignal(nd_audio, int(sample_rate))
                audio.normalize()
                # store original sample rate
                self.raw_audios.append(audio)
                # if we're not down sampling we need to fill the self._sampled_audios list explicitly
                if not self.down_sampling:
                    self._sampled_audios.append(audio)
                self.num_of_files += 1
            except IOError as e:
                print('Could not read:', audio_file, ':', e, '- it\'s ok, skipping.')
        print("%d files loaded" % self.num_of_files)

    def down_sampling(self):

        self.new_sample_rate = 2 * int(self.def_highest_freq) + self._offset
        # compute factor for down sampling
        q = config.frequency_of_format / self.new_sample_rate
        print("Original sample rate %d, new sample rate %d, down sampling factor %d" % (config.frequency_of_format,
                                                                                        self.new_sample_rate,
                                                                                        q))
        for i, raw_audio in enumerate(self.raw_audios):
            # use scipy.signal.decimate to down sample
            new_audio = AudioSignal(sg.decimate(raw_audio.nd_signal, q), self.new_sample_rate)
            self._sampled_audios.append(new_audio)
            print("(old/new) shape ", self.raw_audios[i].nd_signal.shape, self._sampled_audios[i].nd_signal.shape)

    def create_train_matrix(self, f_name_out=None):
        """
            This code is a modification of some of the data loading utilities of GRUV repository
             https://github.com/MattVitelli/GRUV/
        """
        # block sizes used for training - this defines the size of our input state
        self.block_size = self.new_sample_rate / self.chunks_per_sec
        # Used later for zero-padding song sequences
        max_seq_len = int(round((self.new_sample_rate * self._clip_length) / self.block_size))
        print("Using new sample rate %d and block size %d, max seq length %d" % (self.new_sample_rate, self.block_size,
                                                                                 max_seq_len))
        chunks_X = []
        chunks_Y = []

        for idx, audio in enumerate(self._sampled_audios):
            x_t = convert_nd_audio_to_sample_blocks(audio.nd_signal, self.block_size)
            y_t = x_t[1:]
            y_t.append(np.zeros(self.block_size))  # Add special end block composed of all zeros

            cur_seq = 0
            total_seq = len(x_t)
            # print("total_seq ", total_seq)
            # print("max_seq_len ", max_seq_len)
            while cur_seq + max_seq_len < total_seq:
                chunks_X.append(x_t[cur_seq:cur_seq + max_seq_len])
                chunks_Y.append(y_t[cur_seq:cur_seq + max_seq_len])
                cur_seq += max_seq_len

        num_examples = len(chunks_X)
        num_dims_out = self.block_size
        print("num examples %d" % num_examples)
        out_shape = (num_examples, max_seq_len, num_dims_out)
        x_data = np.zeros(out_shape)
        y_data = np.zeros(out_shape)
        for n in xrange(num_examples):
            for i in xrange(max_seq_len):
                x_data[n][i] = chunks_X[n][i]
                y_data[n][i] = chunks_Y[n][i]

        print("x_data shape", x_data.shape)
        mean_x = np.mean(np.mean(x_data, axis=0), axis=0)  # Mean across num examples and num time steps
        # STD across num examples and num timesteps
        std_x = np.sqrt(np.mean(np.mean(np.abs(x_data - mean_x) ** 2, axis=0), axis=0))
        std_x = np.maximum(1.0e-8, std_x)  # Clamp variance if too tiny
        x_data[:][:] -= mean_x
        x_data[:][:] /= std_x
        y_data[:][:] -= mean_x
        y_data[:][:] /= std_x

        self._train_signal_pairs = {'x_data': x_data, 'y_data': y_data}
        self.signal_mean_std = {'mean_x': mean_x, 'std_x': std_x}

        if f_name_out is not None:
            numpy_file = self._root_path + f_name_out + '.npy'
            print('Save to disk (%s)...' % numpy_file)

            with open(numpy_file, 'wb') as fs:
                np.savez_compressed(fs, **self._train_signal_pairs)
            # we need the mean and stddev when reconstructing the signal
            numpy_file = self._root_path + f_name_out + '_stats.npy'

            with open(numpy_file, 'wb') as fs:
                np.savez_compressed(fs, **self.signal_mean_std)

    @property
    def train_signal_pairs(self):
        if self._train_signal_pairs is None:
            print("signal pairs don't exist yet. creating default.")
            self.create_train_matrix()
        return self._train_signal_pairs

    @property
    def train_spectra_pairs(self):
        if self._train_spectra_pairs is None:
            print("spectra pairs don't exist yet. creating default.")
            self.create_train_spectra()
        return self._train_spectra_pairs

    def create_train_spectra(self, f_name_out=None):
        x_signal = self.train_signal_pairs['x_data']
        sign_len = x_signal.shape[-1]
        spec_res = sign_len/2
        x_spectra = (fft(x_signal) / sign_len)[:, :, :spec_res]
        x_spectra = np.concatenate((np.real(x_spectra), np.imag(x_spectra)), axis=2)
        filler = np.zeros((x_spectra.shape[0], 1, x_spectra.shape[2]))
        y_spectra = np.concatenate((x_spectra[:, 1:, :], filler), axis=1)
        self._train_spectra_pairs = {'x_data': x_spectra, 'y_data': y_spectra}

        if f_name_out is not None:
            numpy_file = self._root_path + f_name_out + '.npy'
            print('Save to disk (%s)...' % numpy_file)

            with open(numpy_file, 'wb') as fs:
                np.savez_compressed(fs, **self._train_spectra_pairs)


    def train_batches(self, a_type='sampled', batch_size=None):
        idx = 0
        while True:
            if batch_size == idx + 1 or idx == self.num_of_files:
                break
            print("BatchID %d of type %s" % (idx, a_type))
            if a_type == 'raw':
                yield self.raw_audios[idx]
            else:
                yield self._sampled_audios[idx]
            idx += 1


class AudioSignal(object):

    def __init__(self, nd_signal, sample_rate):
        self.nd_signal = nd_signal
        self.sample_rate = sample_rate
        self.duration = nd_signal.shape[0] / sample_rate
        self.is_normalized = False
        self._spectra = None

    def make_matrix(self):
        return self.nd_signal.reshape((self.duration, self.sample_rate))

    def normalize(self):
        # normalize amplitude values to a range between -1 and 1
        self.is_normalized = True
        self.nd_signal = self.nd_signal / 32767.0 # divide by max number for 16-bit encoding

    @property
    def normalized_signal_matrix(self):
        if not self.is_normalized:
            self.normalize()
        return self.make_matrix()

    def divisible_matrix(self, divisor):
        rest = self.sample_rate % divisor
        return self.normalized_signal_matrix[:, rest:]

    def custom_matrix(self):
        dim0 = self.duration * self.chunks_per_sec
        dim1 = self.nd_signal.shape[0] / dim0
        sig_len = dim0 * dim1
        print(sig_len)
        return np.reshape(self.nd_signal[0:sig_len], (dim0, dim1))


def plot_signal_simple(sig, t_range=None, p_title=None):
    # plot a range of the signal
    if t_range is None:
        if len(sig) > 1e5:
            t_range = 100000
        else:
            t_range = -1

    if p_title is not None:
        plt.title(p_title)
    plt.xlabel("time")
    plt.ylabel("frequency")
    plt.plot(sig[0:t_range])
    plt.show()


# myAudios = AudioPipeline('instrument_samples/flute_nonvib_wav', 10, highest_freq=5000, clip_len=2, chunks_per_sec=4)
# print(myAudios.train_spectra_pairs.keys())
# x = myAudios.train_spectra_pairs['x_data']
# y = myAudios.train_spectra_pairs['y_data']
# print(x.shape)
# print(y.shape)
# d = myAudios.train_signal_pairs['x_data']
# print(d.shape)
# load 2 audio files
# batches = myAudios.train_batches()
# audio = next(batches)
# print("Shape ", audio.custom_matrix(1).shape)
# print(next(batches))
# print(next(batches))
