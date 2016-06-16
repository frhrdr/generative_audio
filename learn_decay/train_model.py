
from __future__ import print_function
from lstm_utils import create_lstm_network
from audio_preprocessing.pipeline import load_matrix, AudioPipeline
from audio_preprocessing.cconfig import config


data = 'cello_arco_train'
folder_spec = 'cello_arco_train/'
data = 'train_flute'
folder_spec = 'D - data_flute_vib/'
max_instru_freq = 8000 # cello
max_instru_freq = 5000
# get trainings data
myAudios = AudioPipeline(folder_spec, 90, max_instru_freq, 2, chunks_per_sec=4)
myAudios.create_train_matrix(data)

x_data, y_data = load_matrix(folder_spec, data)

num_frequency_dimensions = x_data.shape[2]
num_hidden_dimensions = 1024
batch_size = 10
epochs = 100

model = create_lstm_network(num_frequency_dimensions, num_hidden_dimensions)
print(model.summary())
print('Start Training')
model.fit(x_data, y_data, batch_size=batch_size, nb_epoch=epochs, verbose=1, validation_split=0.0)

print('Training complete')
model_output = config.datapath + data + '_weights'
model.save_weights(model_output, overwrite=True)