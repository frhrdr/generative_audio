from train_func import train_func
from generate_seq import gen_seq_full

"""
    First note that the script assumes the following directory structure:
    <root of repo>
            data/
                gen_samples/
                instrument_samples/
                                    sub-directory for each instrument e.g. "guitar_train/" and "guitar_test/"
                weight_matrices/
                          

    this script can be used to train and generate one of the following RNN models
    (1) architecture='1': simple LSTM model
        
    (2) architecture='2': CNN combined with LSTM model
    
    Some explanations in regards to the parameters/variables:
    
        (1) train_dir: points to a directory that contains the wav sound files the model is being trained on.
                       The directory must exist under the root of the repository in the "data/instrument_samples/" 
                       directory
        (2) train:     (a) if set to "False" the model will be loaded together with the weights in order to generate
                       (default) two sound signals based on prime signals that reside in the directory specified 
                       by the "folder_spec" variable.
                       (b) if set to "True" the model will be first trained in addition to what is described under (a)
                           The program will search for compressed numpy files in the "train_dir" which contains
                           the necessary training data (matrices). Otherwise the wav files in the directory will be
                           loaded, preprocessed and saved. The naming convention is:
                           e.g.: guitar_train_45files_5sec_60res_1400maxf_spec.npy
        (3) data:       specifies the name of the file that contains the training data (see above for more explanation)                   
                       
        (4) model_name: when "train" is set to "True this variable will be automatically set.
                        when "train" is set to "False" it is important that the variable specifies the "root" name
                        of the model and weight matrix file.
                        E.g. if model_name =  'guitar_train_45files_5sec_60res_1400maxf_spec_m1_512hid_1lyrs_180ep_linearact'
                             the program expects that the following files are present in the "data/weight_matrices/" 
                             directory:
                             guitar_train_45files_5sec_60res_1400maxf_spec_m1_512hid_1lyrs_180ep_linearact_model.json
                             guitar_train_45files_5sec_60res_1400maxf_spec_m1_512hid_1lyrs_180ep_linearact_weights.h5
        (5) prime_length: number of time slices used for the prime signal. E.g. the above mentioned model expects
                          sound signals that are separated into 60 time slices/second. 
                          We found that if we use 1/3 of a second (in this case prime_length=20) as prime this results 
                          most oftnen in reasonable results while generating new sequences.
        (6) num_of_tests: the number of times the program will generate a test sequence
"""

train = True
if train:
    train_dir = 'guitar_train/'
    _, w_mat_name, d_mat_name = train_func(train_dir,
                                           n_hid_neurons=512,
                                           n_rec_layers=1,
                                           epochs=180,              
                                           highest_freq=1400,       # highest frequency of the specific instrument
                                           n_to_load=45,            # number of wav files to load for training
                                           down_sampling=True,
                                           save_weights=True,
                                           chunks_per_sec=90,       # number of time slices per second
                                           clip_len=5,              # clip/padd the sound signal to this length in seconds
                                           add_spectra=False,
                                           architecture='1',
                                           mean_std_per_file=True,  # True=calc global mean/stddev; False=calc mean/stddev for each file
                                           activation='linear')     # activation function used in dense layers

    folder_spec = '/instrument_samples/guitar_train/'   # used for the generation process, can point to directory of test files
    data = d_mat_name
    model_name = w_mat_name
else:
    model_name = 'guitar_train_45files_5sec_40res_1400maxf_spec_m1_2048hid_1lyrs_100ep_linearact'
    data = 'guitar_train_45files_5sec_40res_1400maxf_spec'

    folder_spec = '/instrument_samples/guitar_train/'

prime_length = 20
num_of_tests = 2
gen_seq_full(folder_spec=folder_spec, data=data, model_name=model_name,
             prime_length=prime_length, num_of_tests=num_of_tests, add_spectra=False, mean_std_per_file=True)
