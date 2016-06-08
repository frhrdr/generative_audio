from audio_preprocessing.pipeline import AudioPipeline
import numpy as np
import matplotlib.pyplot as plt

myAudios = AudioPipeline(1)

batches = myAudios.train_batches()
# print(next(batches))
x_data = next(batches).divisible_matrix(16)


train_audio = x_data[1, :]

ans = np.fft.fft(train_audio)

print ans.shape
print ans[ :20]

plt.plot(np.real(ans[ :]), 'g')
plt.plot(np.imag(ans[ :]), 'r')
plt.plot(x_data[1, :])
plt.show()

