from __future__ import print_function
import numpy as np
import matplotlib as plt
import os
import scipy.io.wavfile as wav


I_FOLDER = "/home/jogi/Dropbox/Study/030 - Artificial Intelligence/P - Projects/010 - AI project 2016/D - data/"

class AudioPipeline(object):

    def __init__(self, folder):
        self._raw_audios = []
        self._folder = folder
        self._num_of_files = 0
        self._next_generator = none

    def load_data(self):
        audio_files = os.listdir(I_FOLDER)
        for audio_index, audio in enumerate(audio_files):
            audio_file = os.path.join(self._folder, audio)
            try:
                nd_audio = wav.read(audio_file)
                self._raw_audios.append(nd_audio)
                self._num_of_files += self._num_of_files
            except IOError as e:
                print('Could not read:', audio_file, ':', e, '- it\'s ok, skipping.')
        print("%d files loaded" % self._num_of_files)


    def nextFile_generator(self, max_files=None):

        for i in range(self._num_of_files):
            yield self._raw_audios[i]

    def next_audio(self):


        return next(self._batch_generator)



myAudio = AudioPipeline(I_FOLDER)
myAudio.load_data()