from __future__ import print_function
from pylab import plot, show, title, xlabel, ylabel, subplot
import numpy as np
from scipy import fft, arange
from audio_preprocessing.pipeline import AudioPipeline


def rms_error(signal, reconstruction, verbose=False):
    n = len(reconstruction)
    rmse = np.linalg.norm(reconstruction - signal) / np.sqrt(n)
    if verbose:
        print("RMS Error: ", rmse)
    return rmse


def plot_signals(signal, reconstruction, separate=False, display=True):
    t = range(len(signal))

    if separate:
        subplot(2, 1, 1)
        title('signal')
        plot(t, signal)
        xlabel('Time')
        ylabel('Amplitude')
        subplot(2, 1, 2)
        title('reconstruction')
        plot(t, reconstruction)
        xlabel('Time')
        ylabel('Amplitude')
    else:
        title('signal: g, reconstruction: b')
        plot(t, signal, "g")
        plot(t, reconstruction, "b")
        xlabel('Time')
        ylabel('Amplitude')
    if display:
        show()


def plot_spectra(signal, reconstruction, sampling_freq, separate=False, display=True):
    n = len(signal)
    k = arange(n)
    T = n/float(sampling_freq)
    frq = (k/T)[range(n/2)] # one side frequency range

    sig_spec = np.abs(fft(signal)/n)[range(n/2)]
    rec_spec = np.abs(fft(reconstruction)/n)[range(n/2)]

    if separate:
        subplot(2, 1, 1)
        title('signal')
        plot(frq, sig_spec)
        xlabel('Freq (Hz)')
        ylabel('|Y(freq)|')
        subplot(2, 1, 2)
        title('reconstruction')
        plot(frq, rec_spec)
        xlabel('Freq (Hz)')
        ylabel('|Y(freq)|')
    else:
        title('signal: g, reconstruction: b')
        plot(frq, sig_spec, "g")
        plot(frq, rec_spec, "b")
        xlabel('Freq (Hz)')
        ylabel('|Y(freq)|')
    if display:
        show()


# myAudios = AudioPipeline('instrument_samples/flute_nonvib_wav', 10, highest_freq=5000, clip_len=2, chunks_per_sec=4)
#
# d = myAudios.train_signal_pairs['x_data']
# sig1 = d[1, 2, :]
# sig2 = d[1, 3, :]
# sampling_freq = 44100
#
# rms_error(sig1, sig2, verbose=False)
# plot_signals(sig1, sig2, separate=True, display=True)
# plot_spectra(sig1, sig2, sampling_freq, separate=True, display=True)