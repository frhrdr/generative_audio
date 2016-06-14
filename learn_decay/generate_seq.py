from __future__ import print_function
from audio_preprocessing.cconfig import config
from audio_preprocessing.pipeline import load_matrix
import numpy as np
import scipy.io.wavfile as wav
from keras.models import model_from_json


def write_np_as_wav(X, sample_rate, filename):

    Xnew = X.astype('int16')
    if ".wav" not in filename:
        filename += ".wav"
    wav.write(filename, sample_rate, Xnew)


def generate_prime_sequence(t_data, seq_length=3, index=-1):
    # dim0 contains number of trainings examples, randomly choose one example
    if index == -1:
        # random sample a sound
        example = np.random.randint(t_data.shape[0], size=1)[0]
    else:
        # get a specific sound
        example = t_data[index]

    begin_seq = t_data[example, 0:seq_length, :]
    total_seq = t_data[example, :, :]
    return np.reshape(begin_seq, (1, begin_seq.shape[0], begin_seq.shape[1])), \
            np.reshape(total_seq, (1, total_seq.shape[0], total_seq.shape[1]))


def denormalize_signal(signal, mean_s, stddev_s, max_amplitude=32767.0):
    signal = signal + mean_s
    signal = signal * stddev_s
    signal = signal * max_amplitude
    return signal.astype('int16')


def get_model(l_model_name, print_sum=False):

    model_input_file = config.datapath + '/weight_matrices/' + l_model_name + "_model.json"
    model_weights_input_file = config.datapath + '/weight_matrices/' + l_model_name + "_weights.h5"

    l_model = model_from_json(model_input_file)
    # load model weights
    l_model.load_weights(model_weights_input_file)
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


# some parameters

folder_spec = 'instrument_samples/cello_train/'
data = 'cello_train_2files_4res_8000maxf'
model_name = 'cello_train_2files_4res_8000maxf_1024hid_1ep'
# General part that only needs to be executed once for generating
# multiple sequences:
# (1) load/get test sound signals
# (2) load/get statistics (mean/standard deviation) to reconstruct unnormalized sound signal
# (3) load model and weights determined in earlier training phase
x_test, y_test = load_matrix(folder_spec, data) # f_name contains the physical file names
# total length of the signal to be generated
sequence_len = x_test.shape[1]
# add mean and stddev to signal
mean_x, stddev_x, f_name = load_matrix(folder_spec, data + "_stats")
model = get_model(model_name, print_sum=True)

# This part needs to be placed in a loop when generating more than one sequence
print("Get %s sound as prime sequence " % f_name[5])
sequence_begin, sequence_total = generate_prime_sequence(x_test, seq_length=3, index=5)
generated_sequence = generate_sequence(model, sequence_begin, sequence_len, mean_x, stddev_x)

sequence_total = denormalize_signal(sequence_total, mean_x, stddev_x)
sequence_total = np.reshape(sequence_total, sequence_total.shape[1] * sequence_total.shape[2])

# plot_signals(sequence_total, generated_sequence, True)

gen_filename = 'cello_example2'
write_np_as_wav(generated_sequence, config.frequency_of_format, gen_filename)

