from __future__ import print_function
from audio_preprocessing.cconfig import config
from lstm_utils import create_lstm_network
from audio_preprocessing.pipeline import load_matrix
import numpy as np
import scipy.io.wavfile as wav
from learn_decay.signal_comparisons import plot_signals
import matplotlib.pyplot as plt


# some parameters
data = 'cello_train'
folder_spec = 'cello_train/'
data = 'train_flute'
folder_spec = 'D - data_flute_vib/'


def write_np_as_wav(X, sample_rate, filename):

    Xnew = X.astype('int16')
    if ".wav" not in filename:
        filename += ".wav"
    wav.write(filename, sample_rate, Xnew)


def generate_prime_sequence(t_data, seq_length=3):
    # dim0 contains number of trainings examples, randomly choose one example
    example = np.random.randint(t_data.shape[0], size=1)[0]
    begin_seq = t_data[example, 0:seq_length, :]
    total_seq = t_data[example, :, :]
    return np.reshape(begin_seq, (1, begin_seq.shape[0], begin_seq.shape[1])), \
            np.reshape(total_seq, (1, total_seq.shape[0], total_seq.shape[1]))


def denormalize_signal(signal, mean_s, stddev_s, max_amplitude=32767.0):
    signal = signal + mean_s
    signal = signal * stddev_s
    signal = signal * max_amplitude
    return signal.astype('int16')


def get_model(num_freq_dim,  num_hidden_dimensions=1024, use_stateful=False, print_sum=False):

    model_path = config.datapath + data + '_weights'
    if use_stateful:
        l_model = create_lstm_network(num_freq_dim, num_hidden_dimensions, stateful=True)
    else:
        l_model = create_lstm_network(num_freq_dim, num_hidden_dimensions)
    # load model weights
    l_model.load_weights(model_path)
    if print_sum:
        l_model.summary()
    return l_model


def generate_sequence(model, prime_seq, sequence_len, mean_s, stddev_s, use_stateful=False):
    """

    :param model:
    :param prime_seq:
    :param sequence_len: total length of the signal to generate
    :param use_stateful
    :return:
    """
    prime = prime_seq.copy()

    generated_seq = np.zeros((sequence_len, prime.shape[2]))
    #     The prime sequence contains e.g. [x1, x2, x3]
    # (1) The first step is to copy the "prime" sequence into the new generated sequence
    generated_seq[:prime.shape[1]] = prime[0][:]
    # it should be possible to use stateful recursions.
    # abandoned for now, because the efficiency is probably not worth the effort
    if use_stateful:
        for idx in range(sequence_len - prime.shape[1]):
            predict_seq = model.predict(prime)
            # after initial priming, only put in one time-slice at a time
            prime = predict_seq[-1]
    else:
        #
        # (2) In each step we take the last time slice x_t+1 and concatenate it to the original prime signal
        #     e.g.  Prime                           Output                  Generated sequence
        #           [x1, x2, x3]                    [x2', x3', x4']         [x1, x2, x3, x4']
        #           [x1, x2, x3, x4']               [x2', x3', x4', x5']    [x1, x2, x3, x4', x5]
        for idx in range(sequence_len - prime.shape[1]):
            print("prime ", prime.shape)
            predict_seq = model.predict(prime)
            print("predict_seq ", predict_seq.shape)
            seq_t_plus_1 = predict_seq[0, predict_seq.shape[1] - 1, :]
            generated_seq[predict_seq.shape[1]] = seq_t_plus_1
            seq_t_plus_1 = np.reshape(seq_t_plus_1, (1, 1, seq_t_plus_1.shape[0]))
            prime = np.concatenate((prime, seq_t_plus_1), axis=1)

        # tmp = np.reshape(generated_seq, generated_seq.shape[0] * generated_seq.shape[1])
        # fig = plt.figure(1)
        # plt.plot(tmp)
        # plt.title("before normalization")
        # fig = plt.figure(2)
        generated_seq = denormalize_signal(generated_seq, mean_s, stddev_s)
        # tmp = np.reshape(tmp, generated_seq.shape[0] * generated_seq.shape[1])
        # plt.title("after normalization")
        # plt.plot(tmp)
        # plt.show()
        return np.reshape(generated_seq, generated_seq.shape[0] * generated_seq.shape[1])


x_test, y_test = load_matrix(folder_spec, data)
# total length of the signal to be generated
sequence_len = x_test.shape[1]
# add mean and stddev to signal
mean_x, stddev_x = load_matrix(folder_spec, data + "_stats")
sequence_begin, sequence_total = generate_prime_sequence(x_test, 3)


num_frequency_dimensions = x_test.shape[2]
model = get_model(num_frequency_dimensions, 1024, print_sum=True)
generated_sequence = generate_sequence(model, sequence_begin, sequence_len, mean_x, stddev_x)

sequence_total = denormalize_signal(sequence_total, mean_x, stddev_x)
sequence_total = np.reshape(sequence_total, sequence_total.shape[1] * sequence_total.shape[2])

# plot_signals(sequence_total, generated_sequence, True)

gen_filename = 'cello_example2'
write_np_as_wav(generated_sequence, config.frequency_of_format, gen_filename)

