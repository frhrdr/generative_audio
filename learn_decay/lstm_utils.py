from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Flatten, Reshape
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D
from keras.layers import BatchNormalization, MaxPooling1D, UpSampling1D
import keras.backend as K


# taken from MattVitelli's GRUV project
# https://github.com/MattVitelli/GRUV/blob/master/nn_utils/network_utils.py
def create_lstm_network(num_frequency_dimensions, num_hidden_dimensions,
                        num_recurrent_units=1, stateful=False, l_activation='linear'):

    model = Sequential()
    # This layer converts frequency space to hidden space
    model.add(TimeDistributed(Dense(num_hidden_dimensions, activation=l_activation),
                              input_shape=(None, num_frequency_dimensions)))
    # model.add(TimeDistributedDense(input_dim=num_frequency_dimensions, output_dim=num_hidden_dimensions))
    for cur_unit in xrange(num_recurrent_units):
        model.add(LSTM(num_hidden_dimensions, return_sequences=True, stateful=stateful))

    # This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(input_dim=num_hidden_dimensions, output_dim=num_frequency_dimensions,
                                    activation=l_activation)))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model


def create_conv_lstm_network(num_frequency_dimensions, num_hidden_dimensions,
                                            num_recurrent_units=1, stateful=False):

    model = Sequential()
    model.add(TimeDistributed(Convolution1D(32, 41, border_mode='same', name="1_conv1d", activation='tanh'),
                              input_shape=(None, num_frequency_dimensions, 1), name="1_timedist"))
    # model.add(TimeDistributed(BatchNormalization()))
    # model.add(TimeDistributed(MaxPooling1D(2, border_mode='valid')))
    # model.add(TimeDistributed(Convolution1D(16, 17, activation='tanh', border_mode='same')))
    # model.add(TimeDistributed(BatchNormalization()))
    # model.add(TimeDistributed(MaxPooling1D(2, border_mode='valid')))
    # model.add(TimeDistributed(Convolution1D(8, 3, activation='tanh', border_mode='same')))
    # model.add(TimeDistributed(BatchNormalization()))
    # model.add(TimeDistributed(MaxPooling1D(2, border_mode='valid')))
    model.add(TimeDistributed(Flatten()))
    for cur_unit in xrange(num_recurrent_units):
        model.add(LSTM(output_dim=num_hidden_dimensions, return_sequences=True, stateful=stateful))
    model.add(TimeDistributed(Dense(input_dim=num_hidden_dimensions, output_dim=num_frequency_dimensions)))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model


def create_autoconv_lstm_network(num_frequency_dimensions, num_hidden_dimensions,
                                 num_recurrent_units=1, stateful=False):

    model = Sequential()
    model.add(TimeDistributed(Convolution1D(32, 41, border_mode='same', name="1_conv1d", activation='relu'),
                              input_shape=(None, num_frequency_dimensions, 1), name="1_timedist"))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling1D(2, border_mode='valid')))
    model.add(TimeDistributed(Convolution1D(16, 17, activation='relu', border_mode='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling1D(2, border_mode='valid')))
    model.add(TimeDistributed(Convolution1D(8, 3, activation='relu', border_mode='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling1D(2, border_mode='valid')))
    last_2d_shape = model.output_shape[-2:]
    output_num_rnn = last_2d_shape[0] * last_2d_shape[1]

    model.add(TimeDistributed(Flatten()))

    for cur_unit in xrange(num_recurrent_units):
        model.add(LSTM(output_dim=num_hidden_dimensions, return_sequences=True, stateful=stateful))
    model.add(TimeDistributed(Dense(input_dim=num_hidden_dimensions, output_dim=output_num_rnn)))

    model.add(TimeDistributed(Reshape(last_2d_shape)))
    model.add(TimeDistributed(Convolution1D(8, 3, activation='relu', border_mode='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(UpSampling1D(2)))
    model.add(TimeDistributed(Convolution1D(16, 17, activation='relu', border_mode='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(UpSampling1D(2)))
    model.add(TimeDistributed(Convolution1D(32, 41, activation='relu', border_mode='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(UpSampling1D(2)))
    model.add(TimeDistributed(Convolution1D(1, 3, activation='tanh', border_mode='same')))
    last_2d_shape = model.output_shape[-2]
    model.add(TimeDistributed(Reshape((last_2d_shape,))))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')

    return model