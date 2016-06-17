from __future__ import print_function
from train_func import train_func

train_dir = 'guitar_train'
_, w_mat_name = train_func(train_dir,
                           n_hid_neurons=1024,
                           n_rec_layers=1,
                           epochs=150,
                           highest_freq=1400,
                           n_to_load=45,
                           down_sampling=True,
                           save_weights=True,
                           chunks_per_sec=10,
                           clip_len=4)

print("Saved weights to %s" % w_mat_name)



