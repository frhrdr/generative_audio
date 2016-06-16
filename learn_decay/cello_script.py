from __future__ import print_function
from train_func import train_func

train_dir = 'guitar_train'
_, w_mat_name = train_func(train_dir,
                            n_hid=1024,
                            n_recur=1,
                            epochs=100,
                            highest_freq=8000,
                            n_to_load=1,
                            down_sampling=False,
                            save_weights=True,
                            chunks_per_sec=10,
                            clip_len=2)

print("Saved weights to %s" % w_mat_name)



