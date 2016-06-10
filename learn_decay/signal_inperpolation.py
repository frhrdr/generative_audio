from scipy.signal import resample
from audio_preprocessing.pipeline import AudioPipeline
from learn_decay.signal_comparisons import plot_signals


def interpolate_signal(signal, sig_freq, target_freq=44100.0):
    source_samples = len(signal)
    target_samples = source_samples * target_freq / float(sig_freq)
    return resample(signal, target_samples)

subAudios = AudioPipeline('instrument_samples/flute_nonvib_wav', 1, down_sampling=True)
rawAudios = AudioPipeline('instrument_samples/flute_nonvib_wav', 1)
d1 = rawAudios.train_signal_pairs['x_data']
sig1 = d1[1, 2, :]
d2 = subAudios.train_signal_pairs['x_data']
sig2 = d2[1, 3, :]
sig2 = interpolate_signal(sig2, subAudios.def_highest_freq)
sampling_freq = 44100
plot_signals(sig1, sig2)
