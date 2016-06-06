
from __future__ import print_function
from cconfig import config
import numpy as np
import os
import scipy.io.wavfile as wav
import scipy.signal as sg
import glob
import matplotlib.pyplot as plt


class AudioPipeline(object):

    def __init__(self, n_to_load=1):
        self.raw_audios = []
        self.num_of_files = 0
        self._sampled_audios = []
        self._root_path = config.datapath
        self.def_highest_freq = 440
        self._high_freqs = []
        self._offset = 2
        self._n_to_load = n_to_load

        self.load_data()
        self.down_sampling()

    def load_data(self):

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
                self.raw_audios.append(audio)
                # try to get an entry for highest frequency, otherwise take default
                self._high_freqs.append(t_dict.get(audio, self.def_highest_freq))
                self.num_of_files += 1
                if self._n_to_load != 0:
                    if self.num_of_files == self._n_to_load:
                        break
            except IOError as e:
                print('Could not read:', audio_file, ':', e, '- it\'s ok, skipping.')
        print("%d files loaded" % self.num_of_files)

    def down_sampling(self):

        for i, raw_audio in enumerate(self.raw_audios):
            new_sample_rate = 2 * int(self._high_freqs[i]) + self._offset
            # compute factor for down sampling
            q = raw_audio.sample_rate / new_sample_rate
            # use scipy.signal.decimate to down sample
            new_audio = AudioSignal(sg.decimate(raw_audio.nd_signal, q), new_sample_rate)
            self._sampled_audios.append(new_audio)
            # reshape to matrix, make sure that each row represents 1 second

            print("(old/new) shape ", self.raw_audios[i].nd_signal.shape, self._sampled_audios[i].nd_signal.shape)

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

    def make_matrix(self):
        return self.nd_signal.reshape((self.duration, self.sample_rate))

    def normalize(self):
        # normalize amplitude values to a range between -1 and 1
        self.is_normalized = True
        self.nd_signal = self.nd_signal / float(np.max(np.abs(self.nd_signal)))

    @property
    def normalized_signal_matrix(self):
        if not self.is_normalized:
            self.normalize()
        return self.make_matrix()

    def divisible_matrix(self, divisor):
        rest = self.sample_rate % divisor
        return self.normalized_signal_matrix[:, rest:]



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


# myAudios = AudioPipeline(2)
# load 2 audio files
# batches = myAudios.train_batches()
# print(next(batches))
# print(next(batches))
# x_train = next(myAudios.next_sample('sampled', 2))
# print(x_train.divisible_matrix(16).shape)
# x_test = next(myAudios.next_sample('raw', 2))
# # x_valid = next(myAudio.next_sample('sampled'))
# M = x_train.normalized_signal_matrix
# print("Shape of final matrix ", M.shape)
#
# plt.plot(myAudio.raw_audios[0].nd_signal[0:1000], 'g')
# plt.show()