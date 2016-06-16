from __future__ import print_function
from audio_preprocessing.pipeline import AudioPipeline
from ConvAutoencoder import ConvAutoencoder


root_to_folder = ""
train_dir = 'sines_toy_data/'
data = train_dir + 'sines_mat'
audios = AudioPipeline(train_dir, n_to_load=3, highest_freq=440,
                           clip_len=7, mat_dirs=None, chunks_per_sec=1,
                           down_sampling=True)

batches = audios.train_batches()
x_data = next(batches).divisible_matrix(16)
print(x_data.shape)

train_audio = x_data[:800, :]
test_audio = x_data[800:, :]

auto = ConvAutoencoder(train_audio, test_audio)

auto.train(50, 10, True)

auto.show()