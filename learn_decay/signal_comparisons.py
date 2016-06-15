from __future__ import print_function
from pylab import plot, show, title, xlabel, ylabel, subplot
import numpy as np
from scipy import fft, arange
import scipy.signal.signaltools as sigtool
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
# note: add diff to plots

def rms_error(signal, reconstruction, verbose=False):
    n = len(reconstruction)
    rmse = np.linalg.norm(reconstruction - signal) / np.sqrt(n)
    if verbose:
        print("RMS Error: ", rmse)
    return rmse


def plot_signals(signal, reconstruction, separate=False, display=True):
    t = range(len(signal))
    lost_bits = len(signal) - len(reconstruction)
    if lost_bits > 0:
        reconstruction = np.append(reconstruction, np.zeros((lost_bits,)))
    diff = signal - reconstruction

    if separate:
        subplot(3, 1, 1)
        title('signal')
        plot(t, signal)
        # xlabel('Time')
        ylabel('Amplitude')
        subplot(3, 1, 2)
        title('reconstruction')
        plot(t, reconstruction)
        # xlabel('Time')
        ylabel('Amplitude')
        subplot(3, 1, 3)
        title('difference')
        plot(t, diff)
        xlabel('Time')
        ylabel('Amplitude')
    else:
        subplot(2, 1, 1)
        title('signal: g, reconstruction: b')
        plot(t, signal, "g")
        plot(t, reconstruction, "b")
        # xlabel('Time')
        ylabel('Amplitude')
        subplot(2, 1, 2)
        title('difference')
        plot(t, diff)
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

    lost_bits = len(signal) - len(reconstruction)
    if lost_bits > 0:
        rec_spec = np.append(reconstruction, np.zeros((lost_bits,)))

    diff_spec = sig_spec - rec_spec
    if separate:
        subplot(3, 1, 1)
        title('signal')
        plot(frq, sig_spec)
        # xlabel('Freq (Hz)')
        ylabel('|Y(freq)|')
        subplot(3, 1, 2)
        title('reconstruction')
        plot(frq, rec_spec)
        # xlabel('Freq (Hz)')
        ylabel('|Y(freq)|')
        subplot(3, 1, 3)
        title('difference')
        plot(frq, diff_spec)
        xlabel('Freq (Hz)')
        ylabel('|Y(freq)|')
    else:
        subplot(2, 1, 1)
        title('signal: g, reconstruction: b')
        plot(frq, sig_spec, "g")
        plot(frq, rec_spec, "b")
        # xlabel('Freq (Hz)')
        ylabel('|Y(freq)|')
        subplot(2, 1, 2)
        title('reconstruction')
        plot(frq, diff_spec)
        xlabel('Freq (Hz)')
        ylabel('|Y(freq)|')
    if display:
        show()


def exp_func(t, a, b, c):
    return a * np.exp(-b * t) + c


def fit_sig_decay(signal, s_filter=51, poly=2):
    """
        returns list with three coefficients that describe the exponential decay curve
        of the signal
    """
    t = np.array(range(signal.shape[0]), dtype=float)
    amplitude_envelope = np.abs(sigtool.hilbert(signal))
    smoothed_envelop = savgol_filter(amplitude_envelope, s_filter, poly)
    coefficients, _ = curve_fit(exp_func, t, smoothed_envelop)
    return coefficients


def plot_decays(signal, reconstruction, coeff_signal, coeff_recon, separate=False, display=True):
    """
        first calculates the data points for the exponential decay curve
        based on the coefficients (parameters).
        the plots the decay of original signal and reconstructed signal
    """
    t = np.array(range(signal.shape[0]), dtype=float)
    s_data = exp_func(t, coeff_signal[0], coeff_signal[1], coeff_signal[2])
    r_data = exp_func(t, coeff_recon[0], coeff_recon[1], coeff_recon[2])

    if separate:
        subplot(2, 1, 1)
        title('signal decay curve')
        plot(t, signal)
        plot(t, s_data, 'r', linewidth=2.0)
        subplot(2, 1, 2)
        title('reconstruction decay curve')
        plot(t, reconstruction)
        plot(t, r_data, 'r', linewidth=2.0)
        xlabel('time')
        ylabel('freq (Hz)')

    else:
        subplot(1, 1, 1)
        title('signal (green), reconstruction (red) decay curve')
        plot(t, signal)
        plot(t, s_data, 'g')
        plot(t, r_data, 'r')
        xlabel('time')
        ylabel('freq (Hz)')

    if display:
        show()

