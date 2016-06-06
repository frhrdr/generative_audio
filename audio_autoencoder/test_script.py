from audio_preprocessing.pipeline import AudioPipeline
from conv_autoencoder import ConvAutoencoder

myAudios = AudioPipeline()
# load 2 audio files
myAudios.load_data(1)
myAudios.down_sampling()
audio_data = next(myAudios.next_sample('sampled'))

train_audio = audio_data[:800, :]
test_audio = audio_data[800:, :]

auto = ConvAutoencoder(train_audio, test_audio)

auto.train(1, 256, True)

