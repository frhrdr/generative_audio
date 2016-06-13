from __future__ import print_function
from audio_preprocessing.cconfig import config
from lstm_utils import create_lstm_network
from audio_preprocessing.pipeline import load_matrix
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav


def write_np_as_wav(X, sample_rate, filename):
    Xnew = X * 32767.0
    Xnew = Xnew.astype('int16')
    if ".wav" not in filename:
        filename += ".wav"
    wav.write(filename, sample_rate, Xnew)


def generate_seq_begin_v1(seq_length, gen_length, data):
    """
        seq_length: length of the sequence that will be used as seed
        gen_length: length of the signal to be generated during prediction
    """
    assert ((seq_length + gen_length) < data.shape[1]), "seed + generate length (%d + %d) can not exceed time dim1 (%d) of data" % \
                                                      (seq_length, gen_length, data.shape[1])
    example = np.random.randint(data.shape[0], size=1)[0]
    # dim1 contains number of time intervals
    timeslice = np.random.randint(1, data.shape[1] - seq_length, size=1)[0]
    begin_seq = data[example, timeslice:timeslice+seq_length, :]
    label_gen_seq = data[example, timeslice+seq_length:timeslice+seq_length+gen_length, :]
    return np.reshape(begin_seq, (1, begin_seq.shape[0], begin_seq.shape[1])), \
           np.reshape(label_gen_seq, (1, label_gen_seq.shape[0], label_gen_seq.shape[1]))


def generate_sequence_begin_v2(seq_length, t_data):
    # dim0 contains number of trainings examples, randomly choose one example
    example = np.random.randint(t_data.shape[0], size=1)[0]
    begin_seq = t_data[example, 0:seq_length, :]
    total_seq = t_data[example, :, :]
    return np.reshape(begin_seq, (1, begin_seq.shape[0], begin_seq.shape[1])), \
            np.reshape(total_seq, (1, total_seq.shape[0], total_seq.shape[1]))

data = 'train_nonvib_flute'
folder_spec = 'D - data_flute_nonvib/'
x_data, y_data = load_matrix(folder_spec, data)
# add mean and stddev to signal
mean_x, stddev_x = load_matrix(folder_spec, data + "_stats")

# sequence_begin, label_gen_seq = generate_seq_begin_v1(3, 2, x_data)
sequence_begin, sequence_total = generate_sequence_begin_v2(3, x_data)
# some parameters
num_time_dimensions = x_data.shape[1]
num_frequency_dimensions = x_data.shape[2]
num_hidden_dimensions = 1024
data = 'train_nonvib_flute'

use_stateful = False  # abandoned for now, because the efficiency is probably not worth the effort

# load model weights
model_path = config.datapath + data + '_weights'
if use_stateful:
    model = create_lstm_network(num_frequency_dimensions, num_hidden_dimensions, stateful=True)
else:
    model = create_lstm_network(num_frequency_dimensions, num_hidden_dimensions)
model.load_weights(model_path)

prime_sequence = sequence_begin.copy()
sequence_len = sequence_total.shape[1]
# we store the generated signal in this variable
# note in the first iteration
generated_seq = np.zeros((sequence_total.shape[1], sequence_total.shape[2]))
#     The prime sequence contains e.g. [x1, x2, x3]
# (1) The first step is to copy the "prime" sequence into the new generated sequence
generated_seq[:prime_sequence.shape[1]] = prime_sequence[0][:]

# it should be possible to use stateful recursions.

if use_stateful:
    for idx in range(sequence_len - prime_sequence.shape[1]):
        predict_seq = model.predict(prime_sequence)
        # after initial priming, only put in one time-slice at a time
        prime_sequence = predict_seq[-1]
else:
    #
    # (2) In each step we take the last time slice x_t+1 and concatenate it to the original prime signal
    #     e.g.  Prime                           Output                  Generated sequence
    #           [x1, x2, x3]                    [x2', x3', x4']         [x1, x2, x3, x4']
    #           [x1, x2, x3, x4']               [x2', x3', x4', x5']    [x1, x2, x3, x4', x5]
    for idx in range(sequence_len - prime_sequence.shape[1]):

        predict_seq = model.predict(prime_sequence)
        print("predict_seq ", predict_seq.shape)
        seq_t_plus_1 = predict_seq[0, predict_seq.shape[1] - 1, :]
        generated_seq[predict_seq.shape[1]] = seq_t_plus_1
        seq_t_plus_1 = np.reshape(seq_t_plus_1, (1, 1, seq_t_plus_1.shape[0]))
        prime_sequence = np.concatenate((prime_sequence, seq_t_plus_1), axis=1)



generated_seq[:][:] += mean_x
generated_seq[:][:] *= stddev_x
generated_seq = np.reshape(generated_seq, generated_seq.shape[0] * generated_seq.shape[1])

write_np_as_wav(generated_seq, config.frequency_of_format, 'flute_example2')

s_rate, x_signal = wav.read('flute_example2.wav')
print(x_signal.shape)
ax1 = plt.subplot(311)
ax1.set_title("Generated signal (with prime)")
plt.plot(x_signal)
ax2 = plt.subplot(312)

print(sequence_total.shape)
sequence_total[:][:] += mean_x
sequence_total[:][:] *= stddev_x
sequence_total = np.reshape(sequence_total, sequence_total.shape[1] * sequence_total.shape[2])
sequence_total = sequence_total * 32767.0
sequence_total = sequence_total.astype('int16')
ax2.set_title("Original signal")
plt.plot(sequence_total)
ax3 = plt.subplot(313)
ax3.set_title("Signal difference")
diff = sequence_total - x_signal
plt.plot(diff)
plt.show()
# diff = sequence_begin - predict_seq
# plt.subplot(411)
# plt.plot(diff[0][0])
# plt.subplot(412)
# plt.plot(diff[0][1])
# plt.subplot(413)
# plt.plot(diff[0][2])
# plt.subplot(414)
# plt.plot(sequence_begin[0][0])
# plt.show()