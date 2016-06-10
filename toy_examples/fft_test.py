from audio_preprocessing.pipeline import AudioPipeline
import numpy as np
from numpy import sin, linspace, pi
from pylab import plot, show, title, xlabel, ylabel, subplot
from scipy import fft, arange

def plotSpectrum(y,Fs):
    """
    Plots a Single-Sided Amplitude Spectrum of y(t)
    """
    n = len(y) # length of the signal
    k = arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range

    Y = fft(y)/n # fft computing and normalization
    Y = Y[range(n/2)]
    print type(Y)
    print abs(Y).dtype
    plot(frq,abs(Y),'r') # plotting the spectrum
    xlabel('Freq (Hz)')
    ylabel('|Y(freq)|')

myAudios = AudioPipeline('instrument_samples/flute_vibrato_wav', 1, 5000, 1)

print myAudios.def_highest_freq
batches = myAudios.train_batches()
# print(next(batches))
signal = next(batches)


train_audio = signal.nd_signal

print 'signal shape ',
print train_audio.shape

Fs = train_audio.shape[0]  # sampling rate
Ts = 1.0/Fs # sampling interval
t = arange(0, 1, Ts) # time vector

print len(t)
print len(train_audio)
subplot(2,1,1)
plot(t,train_audio)
xlabel('Time')
ylabel('Amplitude')
subplot(2, 1, 2)
plotSpectrum(train_audio,Fs)
show()