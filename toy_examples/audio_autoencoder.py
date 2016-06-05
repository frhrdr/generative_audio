from __future__ import print_function

from audio_preprocessing.pipeline import AudioPipeline, plot_signal_simple

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt

# this is the size of our encoded representations
encoding_dim = 40  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(882,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.activity_l1(10e-5))(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(882, activation='tanh')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)

# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))


# First, we'll configure our model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer:
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# get audio files and pre process
myAudios = AudioPipeline()
# load 2 audio files
myAudios.load_data(1)
myAudios.down_sampling()
audio_train = next(myAudios.next_sample('sampled'))

x_train = audio_train.normalized_signal_matrix
# x_test = myAudios.raw_audios[0].normalized_signal_matrix[0:1000]
print(x_train.shape)

autoencoder.fit(x_train, x_train,
                nb_epoch=250,
                batch_size=256,
                shuffle=True,
                validation_data=(x_train, x_train))

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_train)
decoded_imgs = decoder.predict(encoded_imgs)

plt.plot(x_train[0, 0:20], 'b')
plt.plot(decoded_imgs[0, 0:20], 'r')
plt.show()



