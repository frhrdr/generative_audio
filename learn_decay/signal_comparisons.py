from __future__ import print_function
from pylab import plot, show, title, xlabel, ylabel, subplot
import numpy as np
from scipy import fft, arange
import scipy.signal.signaltools as sigtool
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from audio_preprocessing.pipeline import AudioPipeline


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
        ylabel('Amplitude')
        subplot(3, 1, 2)
        title('reconstruction')
        plot(t, reconstruction)
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
        ylabel('|Y(freq)|')
        subplot(3, 1, 2)
        title('reconstruction')
        plot(frq, rec_spec)
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


def spectrogram_from_signal(signal, window_size=1024, stride=512, highest_freq=4000,
                            fname='', display=True, root_signal=False):

    num_specs = (len(signal) - window_size) / stride
    spec = np.zeros((num_specs, window_size/2))
    for idx in range(num_specs):
        # print(idx*stride)
        s = np.abs(np.fft.fft(signal[idx*stride:idx*stride + window_size])[:window_size/2])
        if root_signal:
            s = np.sqrt(s)
        spec[idx, :] = s

    fig, ax = plt.subplots()
    ax.pcolor(spec, cmap=plt.cm.plasma)

    row_tick_stride = spec.shape[1]/10
    col_tick_stride = spec.shape[0]/10

    row_labels = [highest_freq * k / spec.shape[1] for k in range(0, spec.shape[1], row_tick_stride)]
    col_labels = [stride*k for k in range(1, spec.shape[0] + 1, col_tick_stride)]

    ax.set_xticks(np.arange(0, spec.shape[1], row_tick_stride)+0.5, minor=False)
    ax.set_yticks(np.arange(0, spec.shape[0], col_tick_stride)+0.5, minor=False)
    plt.axis('tight')

    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(col_labels, minor=False)
    plt.xlabel('Frequency (in Hz)')
    plt.ylabel('Time (in Frames)')
    if root_signal:
        plt.title(fname + ' (sqrt)')
    else:
        plt.title(fname)

    if display:
        plt.show()


def spectrogram_from_file(directory, file_idx=0, highest_freq=4000, down_sampling=True, display=True, root_signal=False):

    audios = AudioPipeline(directory, n_to_load=file_idx+1, highest_freq=highest_freq, down_sampling=down_sampling)
    fname = audios.files_to_load[file_idx]
    signal = None
    a = audios.train_batches()
    for idx in range(file_idx+1):
        signal = next(a).nd_signal
    spectrogram_from_signal(signal, highest_freq=audios.new_sample_rate/2, display=display,
                            fname=fname, root_signal=root_signal)
