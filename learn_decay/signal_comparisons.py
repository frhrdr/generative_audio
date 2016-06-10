from __future__ import print_function
from pylab import plot, show, title, xlabel, ylabel, subplot
import numpy as np
from scipy import fft, arange

# note: add diff to plots

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


def plot_spectra(signal, reconstruction, sampling_freq, separate=False, display=True, diff=False):
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


