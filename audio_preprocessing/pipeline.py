
from __future__ import print_function
from cconfig import config
import numpy as np
import os
import scipy.io.wavfile as wav
import scipy.signal as sg
import glob


class AudioPipeline(object):

    def __init__(self):
        self._raw_audios = []
        self._raw_sample_rates = []
        self._new_sample_rates = []
        self._num_of_files = 0
        self._sampled_audios = []
        self._root_path = config.datapath
        self.def_highest_freq = 440
        self._high_freqs = []
        self._offset = 2

    def load_data(self, max_files=0):

        # loading highest frequencies per file from "PYTHON_FREQ_FILENAME" file
        freq_input = os.path.join(self._root_path, config.frequency_file)
        dict_content = open(freq_input, 'r').read()
        t_dict = eval(dict_content)

        print("Loading audio files from: %s" % self._root_path)
        os.chdir(self._root_path)
        for audio in glob.glob("*.wav"):
            audio_file = os.path.join(self._root_path, audio)

            try:
                sample_rate, nd_audio = wav.read(audio_file)
                self._raw_sample_rates.append(int(sample_rate))
                self._raw_audios.append(nd_audio)
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

        for i in range(self._num_of_files):
            # compute Nyquist sampling rate for this audio file
            self._new_sample_rates.append(2 * int(self._high_freqs[i]) + self._offset)
            # compute factor for down sampling
            q = self._raw_sample_rates[i] / self._new_sample_rates[i]
            # use scipy.signal.decimate to down sample
            self._sampled_audios.append(sg.decimate(self._raw_audios[i], q))
            # reshape to matrix, make sure that each row represents 1 second
            length_audio_secs = self._raw_audios[i].shape[0] / self._raw_sample_rates[i]
            self._sampled_audios[i] = np.reshape(self._sampled_audios[i],
                                                 (length_audio_secs , self._new_sample_rates[i]))
            print("(old/new) shape ", self._raw_audios[i].shape, self._sampled_audios[i].shape)

    def next_file(self, batch_size=None):

        i = 0
        while True:

            if batch_size == i + 1 or i == self._num_of_files:
                break
            print("nextFileGenerator %d" % i)
            yield self._raw_audios[i]
            i += 1

myAudio = AudioPipeline()
myAudio.load_data(3)
myAudio.down_sampling()

