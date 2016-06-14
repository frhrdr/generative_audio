from scipy.signal import resample
from audio_preprocessing.pipeline import AudioPipeline
from learn_decay.signal_comparisons import plot_signals, plot_spectra
import numpy as np

def interpolate_signal(signal, sig_freq, target_freq=44100.0):
    source_samples = len(signal)
    target_samples = source_samples * target_freq / float(sig_freq)
    print target_samples
    return resample(signal, int(target_samples))


# rawAudios = AudioPipeline('instrument_samples/flute_nonvib_wav', 1)
# d1 = rawAudios.train_signal_pairs['x_data']
# sig1 = d1[0, 2, :]
# print sig1.shape
#
# subAudios = AudioPipeline('instrument_samples/flute_nonvib_wav', 1, down_sampling=True)
# print("file names: ",subAudios.files_to_load)
# d2 = subAudios.train_signal_pairs['x_data']
# sig2 = d2[0, 2, :]
# print sig2.shape
# sig2 = interpolate_signal(sig2, subAudios.new_sample_rate)
# print sig2.shape
# sampling_freq = 44100
#
# plot_signals(sig1, np.append(sig2, np.zeros((3,))), separate=True)
# plot_spectra(sig1, np.append(sig2, np.zeros((3,))), sampling_freq=sampling_freq, separate=True)
