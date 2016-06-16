from train_func import train_func
from generate_seq import gen_seq_full

train_dir = 'cello_pizz_train'
# _, w_mat_name, d_mat_name = train_func(train_dir,
#                                        n_hid=2048,
#                                        n_recur=1,
#                                        epochs=100,
#                                        highest_freq=5000,
#                                        n_to_load=90,
#                                        down_sampling=True,
#                                        save_weights=True,
#                                        chunks_per_sec=30,
#                                        clip_len=4)

folder_spec = 'instrument_samples/cello_pizz_train/'
# data = d_mat_name
# model_name = w_mat_name

model_name = 'cello_pizz_train_90files_30res_5000maxf_2048hid_100ep'
data = 'cello_pizz_train_90files_30res_5000maxf'


prime_length = 2
num_of_tests = 3
gen_seq_full(folder_spec=folder_spec, data=data, model_name=model_name,
             prime_length=prime_length, num_of_tests=num_of_tests)
