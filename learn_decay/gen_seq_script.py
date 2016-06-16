from generate_seq import *

# parameters for generation experiment
folder_spec = 'instrument_samples/cello_arco_train/'
data = 'cello_train_86files_10res_8000maxf'
model_name = 'cello_train_86files_10res_8000maxf_1024hid_200ep'

folder_spec = 'instrument_samples/guitar_train/'
data = 'guitar_train_45files10res1400maxf'
model_name = 'guitar_train_45files10res1400maxf_1024hid_150ep'

# folder_spec = 'instrument_samples/cello_pizz_train/'
# data = 'cello_pizz_train_90files10res8000maxf'
# model_name = 'cello_pizz_train_90files10res8000maxf_1024hid_50ep'

folder_spec = 'instrument_samples/guitar_train/'
data = 'guitar_train_45files10res1400maxf'
model_name = 'guitar_train_45files10res1400maxf_512hid_50ep'


prime_length = 2
num_of_tests = 2

gen_seq_full(folder_spec=folder_spec, data=data, model_name=model_name,
             prime_length=prime_length, num_of_tests=num_of_tests)

