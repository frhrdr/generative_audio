from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM


# taken from MattVitelli's GRUV project
# https://github.com/MattVitelli/GRUV/blob/master/nn_utils/network_utils.py
def create_lstm_network(num_time_dimensions, num_frequency_dimensions, num_hidden_dimensions,
                        num_recurrent_units=1):

    model = Sequential()
    # This layer converts frequency space to hidden space
    model.add(TimeDistributed(Dense(num_hidden_dimensions), input_shape=(num_time_dimensions,
                                                                         num_frequency_dimensions)))

    for cur_unit in xrange(num_recurrent_units):
        model.add(LSTM(num_hidden_dimensions, return_sequences=True))

    # This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(input_dim=num_hidden_dimensions, output_dim=num_frequency_dimensions)))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model
