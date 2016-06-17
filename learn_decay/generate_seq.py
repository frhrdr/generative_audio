from __future__ import print_function
from audio_preprocessing.cconfig import config
from audio_preprocessing.pipeline import load_matrix
import numpy as np
import scipy.io.wavfile as wav
from keras.models import model_from_json
import json
from learn_decay.signal_interpolation import interpolate_signal
from learn_decay.signal_comparisons import plot_signals, plot_spectra, fit_sig_decay, plot_decays
import os


def load_from_file(folder_specs, data, type='data'):
    data_dict = load_matrix(folder_specs, data)
    if type == 'data':
        return data_dict['x_data'], data_dict['y_data']
    elif type == 'stats':
        return data_dict['mean_x'], data_dict['std_x'], data_dict['fnames'], data_dict['sample_rate']
    else:
        print("Not supported")


def write_np_as_wav(signal, sample_rate, filename, show_filename=False):
    # up-sampling if necessary
    if sample_rate != config.frequency_of_format:
        signal = interpolate_signal(signal, sample_rate)
    signal = signal.astype('int16')
    if ".wav" not in filename:
        filename += ".wav"
    if show_filename:
        print("Saving file %s" % filename)
    wav.write(filename, config.frequency_of_format, signal)


def generate_prime_sequence(t_data, seq_length=3, index=-1):
    # dim0 contains number of trainings examples, randomly choose one example
    if index == -1:
        # random sample a sound
        index = np.random.randint(t_data.shape[0], size=1)[0]

    begin_seq = t_data[index, 0:seq_length, :]
    total_seq = t_data[index, :, :]
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
    json_file = open(model_input_file, 'r')
    loaded_model_json = json.load(json_file)
    l_model = model_from_json(loaded_model_json)
    # load model weights
    l_model.load_weights(model_weights_input_file)
    if print_sum:
        l_model.summary()
    return l_model


def generate_sequence(model, prime_seq, sequence_len, mean_s, stddev_s, use_stateful=False, add_spectra=False):
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
    generated_seq[:prime.shape[1]] = prime[0, :, :]
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
        #           [x1, x2, x3, x4']               [x2', x3', x4', x5']    [x1, x2, x3, x4', x5']
        for idx in range(sequence_len - prime_seq.shape[1]):
            # print("prime ", prime.shape)
            predict_seq = model.predict(prime)
            # print("predict_seq ", predict_seq.shape)
            seq_t_plus_1 = predict_seq[0, predict_seq.shape[1] - 1, :]
            generated_seq[predict_seq.shape[1]] = seq_t_plus_1
            seq_t_plus_1 = np.reshape(seq_t_plus_1, (1, 1, seq_t_plus_1.shape[0]))
            prime = np.concatenate((prime, seq_t_plus_1), axis=1)

        if add_spectra:
            spec_cut = generated_seq.shape[1]/2
            generated_seq = generated_seq[:, :spec_cut]

        generated_seq = denormalize_signal(generated_seq, mean_s, stddev_s)
        return np.reshape(generated_seq, generated_seq.shape[0] * generated_seq.shape[1])


def post_processing(orig_signal, gen_signal, sampling_freq, plt_signal=False,
                            plt_spectra=False,
                            plt_decays=False,
                            separate=False,
                            display=True,
                            offset_decay=0):

    if plt_signal:
        plot_signals(orig_signal, gen_signal, separate, display)

    if plt_spectra:
        plot_spectra(orig_signal, gen_signal, sampling_freq, separate, display)
    if plt_decays:
        coeff_signal = fit_sig_decay(orig_signal[offset_decay:], s_filter=51, poly=2)
        # coeff_recon = fit_sig_decay(gen_signal, s_filter=51, poly=2)
        t = np.array(range(orig_signal.shape[0]), dtype=float)
        plot_decays(orig_signal[offset_decay:], gen_signal[offset_decay:],
                    coeff_signal, coeff_signal, separate, display=True)


def gen_seq_full(folder_spec, data, model_name, prime_length, num_of_tests, add_spectra=False):
    # General part that only needs to be executed once for generating
    # multiple sequences:
    # (1) load/get test sound signals
    # (2) load/get statistics (mean/standard deviation) to reconstruct unnormalized sound signal
    # (3) load model and weights determined in earlier training phase
    x_test, _ = load_from_file(folder_spec, data) # f_name contains the physical file names
    if num_of_tests > x_test.shape[0]:
        print("Setting number of tests to maximum %d" % x_test.shape[0])
        num_of_tests = x_test.shape[0]

    # total length of the signal to be generated
    sequence_len = x_test.shape[1]
    block_size = x_test.shape[2]
    # add mean and stddev to signal
    mean_x, stddev_x, f_name, sample_rate = load_from_file(folder_spec, data + "_stats", "stats")
    model = get_model(model_name, print_sum=True)
    gen_directory = config.datapath + "/gen_samples/" + model_name
    if not os.path.exists(gen_directory):
        os.makedirs(gen_directory)

    # This part needs to be placed in a loop when generating more than one sequence
    original_signal = {}
    generated_signal = {}
    for test_index in range(num_of_tests):
        orig_signal_name = f_name[test_index]
        print("Get %s sound as prime sequence " % orig_signal_name)
        sequence_begin, sequence_total = generate_prime_sequence(x_test, seq_length=prime_length, index=test_index)
        generated_sequence = generate_sequence(model, sequence_begin, sequence_len, mean_x, stddev_x,
                                               add_spectra=add_spectra)

        sequence_total = denormalize_signal(sequence_total, mean_x, stddev_x)
        sequence_total = np.reshape(sequence_total, sequence_total.shape[1] * sequence_total.shape[2])
        # save the two signals
        original_signal[orig_signal_name] = sequence_total
        generated_signal[orig_signal_name] = generated_sequence
        # construct new wav file name for generated sequence
        f_parts = orig_signal_name.split('.')
        gen_filename = gen_directory + "/" + ".".join(f_parts[:-1]) + "_gen." + f_parts[-1]
        write_np_as_wav(generated_sequence, sample_rate, gen_filename, True)
        gen_filename = gen_directory + "/" + ".".join(f_parts[:-1]) + "_gen_wp." + f_parts[-1]
        write_np_as_wav(generated_sequence[(prime_length * block_size):], sample_rate, gen_filename, True)
        o_filename = gen_directory + "/" + orig_signal_name
        write_np_as_wav(sequence_total, sample_rate, o_filename, True)

