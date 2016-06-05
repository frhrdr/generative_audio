
from __future__ import print_function
from cconfig import config
import numpy as np
import os
import scipy.io.wavfile as wav
import scipy.signal as sg
import glob
import matplotlib.pyplot as plt


class AudioPipeline(object):

    def __init__(self):
        self._raw_audios = []
        self._num_of_files = 0
        self._sampled_audios = []
        self._root_path = config.datapath
        self.def_highest_freq = 440
        self._high_freqs = []
        self._offset = 2

    def load_data(self, max_files=0):

        # loading highest frequencies per file from "PYTHON_FREQ_FILENAME" file
        if config.frequency_file == "UNUSED":
            t_dict = {}
        else:
            freq_input = os.path.join(self._root_path, config.frequency_file)
            dict_content = open(freq_input, 'r').read()
            t_dict = eval(dict_content)

        print("Loading audio files from: %s" % self._root_path)
        os.chdir(self._root_path)
        for audio in glob.glob("*.wav"):
            audio_file = os.path.join(self._root_path, audio)

            try:
                # read file
                sample_rate, nd_audio = wav.read(audio_file)
                audio = AudioSignal(nd_audio, int(sample_rate))
                audio.normalize()
                # store original sample rate
                self._raw_audios.append(audio)
                # try to get an entry for highest frequency, otherwise take default
                self._high_freqs.append(t_dict.get(audio, self.def_highest_freq))
                self._num_of_files += 1
                if max_files != 0:
                    if self._num_of_files == max_files:
                        break
            except IOError as e:
                print('Could not read:', audio_file, ':', e, '- it\'s ok, skipping.')
        print("%d files loaded" % self._num_of_files)

    def down_sampling(self):

        for i, raw_audio in enumerate(self._raw_audios):
            # compute Nyquist sampling rate for this audio file
            new_sample_rate = 2 * int(self._high_freqs[i]) + self._offset
            # compute factor for down sampling
            q = raw_audio.sample_rate / new_sample_rate
            print("shape raw", raw_audio.nd_signal.shape)
            # use scipy.signal.decimate to down sample
            new_audio = AudioSignal(sg.decimate(raw_audio.nd_signal, q), new_sample_rate)
            self._sampled_audios.append(new_audio)
            # reshape to matrix, make sure that each row represents 1 second

            print("(old/new) shape ", self._raw_audios[i].nd_signal.shape, self._sampled_audios[i].nd_signal.shape)

    def next_sample(self, a_type='raw', batch_size=None):

        i = 0
        while True:

            if batch_size == i + 1 or i == self._num_of_files:
                break
            print("nextFileGenerator %d" % i)
            if a_type == 'raw':
                yield self._raw_audios[i]
            else:
                yield self._sampled_audios[i]
            i += 1


class AudioSignal(object):

    def __init__(self, nd_signal, sample_rate):
        self.nd_signal = nd_signal
        self.sample_rate = sample_rate
        self.duration = nd_signal.shape[0] / sample_rate
        self.is_normalized = False

    def make_matrix(self):
        return self.nd_signal.reshape((self.duration, self.sample_rate))

    def normalize(self):
        # normalize amplitude values to a range between -1 and 1
        self.is_normalized = True
        self.nd_signal /= np.max(np.abs(self.nd_signal), axis=0)

    @property
    def normalized_signal_matrix(self):
        if not self.is_normalized:
            self.normalize()
        return self.make_matrix()


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


#myAudio = AudioPipeline()
# load 2 audio files
#myAudio.load_data(2)
#myAudio.down_sampling()

#x_train = next(myAudio.next_sample('sampled'))
#x_test = next(myAudio.next_sample('sampled'))

#M = x_train.normalized_signal_matrix
#print("Shape of final matrix ", M.shape)
