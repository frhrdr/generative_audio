from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Flatten
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D
from keras.layers import BatchNormalization, MaxPooling1D


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
    model.add(TimeDistributed(Convolution1D(32, 35, border_mode='same', name="1_conv1d", activation='relu'),
                              input_shape=(None, num_frequency_dimensions, 1), name="1_timedist"))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling1D(2, border_mode='valid')))
    model.add(TimeDistributed(Convolution1D(8, 3, border_mode='same', name="2_conv1d", activation='relu'),
                              input_shape=(None, num_frequency_dimensions, 1), name="2_timedist"))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling1D(2, border_mode='valid')))
    model.add(TimeDistributed(Flatten()))

    for cur_unit in xrange(num_recurrent_units):
        model.add(LSTM(output_dim=num_hidden_dimensions, return_sequences=True, stateful=stateful))
    model.add(TimeDistributed(Dense(input_dim=num_hidden_dimensions, output_dim=num_frequency_dimensions)))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model
