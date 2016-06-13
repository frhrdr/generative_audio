import os.path
from audio_preprocessing.cconfig import config
from audio_preprocessing.pipeline import load_matrix, AudioPipeline
from learn_decay.lstm_utils import create_lstm_network

def train_func(train_dir, matrix_file, n_hid, epochs, batch_size):

    file = config.datapath + '/instrument_samples/matrices/' + matrix_file + '.npy'
    # see if matrix file exists
    if not os.path.isfile(file):
        # if not, create it from data
        dir = config.datapath + '/instrument_samples/' + train_dir
        if os.path.isdir(dir):
            audios = AudioPipeline('/instrument_samples/' + train_dir)
            audios.create_train_matrix(f_name_out='/instrument_samples/matrices/' + matrix_file + '.npy')
        else:
            print('both entered paths are invalid. No data loaded')
            return None

    x_data, y_data = load_matrix('/instrument_samples/matrices/', matrix_file)
    num_frequency_dimensions = x_data.shape[2]

    # create model
    model = create_lstm_network(num_frequency_dimensions, n_hid)
    print(model.summary())
    print('Start Training')
    model.fit(x_data, y_data, batch_size=batch_size, nb_epoch=epochs, verbose=1, validation_split=0.0)

    print('Training complete')
    model_output = config.datapath + '/weight_matrices/' + name_from_config(matrix_file, n_hid, epochs)
    model.save_weights(model_output, overwrite=True)


def name_from_config(matrix_file, n_hid, epochs):
    name = matrix_file + '_' + n_hid + 'hid_' + epochs + 'ep'
    return name