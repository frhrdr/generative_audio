from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, TimeDistributedDense
from keras.layers.recurrent import LSTM


# taken from MattVitelli's GRUV project
# https://github.com/MattVitelli/GRUV/blob/master/nn_utils/network_utils.py
def create_lstm_network(num_frequency_dimensions, num_hidden_dimensions,
                        num_recurrent_units=1, stateful=False):

    model = Sequential()
    # This layer converts frequency space to hidden space
    model.add(TimeDistributed(Dense(num_hidden_dimensions), input_shape=(None, num_frequency_dimensions)))
    # model.add(TimeDistributedDense(input_dim=num_frequency_dimensions, output_dim=num_hidden_dimensions))
    for cur_unit in xrange(num_recurrent_units):
        model.add(LSTM(num_hidden_dimensions, return_sequences=True, stateful=stateful))

    # This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(input_dim=num_hidden_dimensions, output_dim=num_frequency_dimensions)))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model
