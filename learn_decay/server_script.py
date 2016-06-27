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
        
    (2) architecture='2': CNN in front of an LSTM model
    
    Some explanations in regards to the parameters/variables:
        (1) train_dir: points to a directory that contains the wav sound files the model is being trained on.
                       The directory must exist under the root of the repository in the "data/instrument_samples/" 
                       directory
        (2) train:     (a) if set to "False" the model will be loaded together with the weights in order to generate
                       (default) two sound signals based on prime signals that reside in the directory specified 
                       by the "folder_spec" variable.
                       (b) if set to "True" the model will be first trained in addition to what is described under (a)
        (3) model_name: when "train" is set to "True this variable will be automatically set.
                        when "train" is set to "False" it is important that the variable specifies the "root" name
                        of the model and weight matrix file.
                        E.g. if model_name =  'guitar_train_45files_5sec_60res_1400maxf_spec_m1_512hid_1lyrs_180ep_linearact'
                             the program expects that the following files are present in the "data/weight_matrices/" 
                             directory:
                             guitar_train_45files_5sec_60res_1400maxf_spec_m1_512hid_1lyrs_180ep_linearact_model.json
                             guitar_train_45files_5sec_60res_1400maxf_spec_m1_512hid_1lyrs_180ep_linearact_weights.h5
                             
"""

train = True
if train:
    train_dir = 'guitar_train/'
    _, w_mat_name, d_mat_name = train_func(train_dir,
                                           n_hid_neurons=512,
                                           n_rec_layers=1,
                                           epochs=180,
                                           highest_freq=1400,
                                           n_to_load=4,
                                           down_sampling=True,
                                           save_weights=True,
                                           chunks_per_sec=90,
                                           clip_len=5,
                                           add_spectra=False,
                                           architecture='1',
                                           mean_std_per_file=True,
                                           activation='linear')

    folder_spec = '/instrument_samples/guitar_train/'
    data = d_mat_name
    model_name = w_mat_name
else:
    model_name = 'guitar_train_45files_5sec_40res_1400maxf_spec_m1_2048hid_1lyrs_100ep_linearact'
    data = 'guitar_train_45files_5sec_40res_1400maxf_spec'

    folder_spec = '/instrument_samples/guitar_train/'

prime_length = 40
num_of_tests = 4
gen_seq_full(folder_spec=folder_spec, data=data, model_name=model_name,
             prime_length=prime_length, num_of_tests=num_of_tests, add_spectra=False, mean_std_per_file=True)
