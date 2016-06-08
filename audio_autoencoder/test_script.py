from audio_preprocessing.pipeline import AudioPipeline
from ConvAutoencoder import ConvAutoencoder

myAudios = AudioPipeline(1)

batches = myAudios.train_batches()
# print(next(batches))
x_data = next(batches).divisible_matrix(16)


train_audio = x_data[:800, :]
test_audio = x_data[800:, :]

auto = ConvAutoencoder(train_audio, test_audio)

auto.train(100, 256, True)

auto.show()