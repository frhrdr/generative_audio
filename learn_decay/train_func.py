import os.path
from audio_preprocessing.cconfig import config
from audio_preprocessing.pipeline import load_matrix, AudioPipeline

def train_func(train_dir, matrix_file, n_hid, epochs, batch_size):

    fname = config.datapath + '/data/instrument_samples/matrices/' + matrix_file + '.npy'
    # see if matrix file exists
    if os.path.isfile(fname):
        load_matrix('/data/instrument_samples/matrices/', matrix_file)
        pass

    dir = config.datapath + '/data/instrument_samples/' + train_dir
    # else load train directory
    if os.path.isdir(dir):
        AudioPipeline()
        pass

    # create model
    model = create_lstm_network(num_frequency_dimensions, num_hidden_dimensions)
    print(model.summary())
    print('Start Training')
    model.fit(x_data, y_data, batch_size=batch_size, nb_epoch=epochs, verbose=1, validation_split=0.0)

    print('Training complete')
    model_output = config.datapath + data + '_weights'
    model.save_weights(model_output, overwrite=True)