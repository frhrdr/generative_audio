from generate_seq import *

# parameters for generation experiment
folder_spec = 'instrument_samples/cello_arco_train/'
data = 'cello_train_86files_10res_8000maxf'
model_name = 'cello_train_86files_10res_8000maxf_1024hid_200ep'

folder_spec = 'instrument_samples/guitar_train/'
data = 'guitar_train_45files10res1400maxf'
model_name = 'guitar_train_45files10res1400maxf_1024hid_150ep'

num_of_tests = 2
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
    sequence_begin, sequence_total = generate_prime_sequence(x_test, seq_length=10, index=test_index)
    generated_sequence = generate_sequence(model, sequence_begin, sequence_len, mean_x, stddev_x)

    sequence_total = denormalize_signal(sequence_total, mean_x, stddev_x)
    sequence_total = np.reshape(sequence_total, sequence_total.shape[1] * sequence_total.shape[2])
    # save the two signals
    original_signal[orig_signal_name] = sequence_total
    generated_signal[orig_signal_name] = generated_sequence
    # construct new wav file name for generated sequence
    f_parts = orig_signal_name.split('.')
    gen_filename = gen_directory + "/" + ".".join(f_parts[:-1]) + "_gen." + f_parts[-1]
    write_np_as_wav(generated_sequence, sample_rate, gen_filename, True)
    o_filename = gen_directory + "/" + orig_signal_name
    write_np_as_wav(sequence_total, sample_rate, o_filename, True)

