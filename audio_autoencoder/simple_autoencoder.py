from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt

class simple_autoencoder:

    def __init__(self, train_set, test_set):
        self.train_data = train_set
        self.test_data = test_set
        self.encoding_dim = train_set.shape[1]

        input_img = Input(shape=(self.encoding_dim,))
        encoded = Dense(self.encoding_dim, activation='relu', activity_regularizer=regularizers.activity_l1(10e-5))(input_img)
        decoded = Dense(self.encoding_dim, activation='sigmoid')(encoded)

        self.autoencoder = Model(input=input_img, output=decoded)

        # this model maps an input to its encoded representation
        self.encoder = Model(input=input_img, output=encoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(self.encoding_dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = self.autoencoder.layers[-1]
        # create the decoder model
        self.decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))


        # First, we'll configure our model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer:
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    def train(self, nb_epoch, batch_size, shuffle):


        self.autoencoder.fit(self.train_data, self.train_data,
                        nb_epoch=nb_epoch,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        validation_data=(self.test_data, self.test_data))



    def show(self):

        # encode and decode some digits
        # note that we take them from the *test* set
        encoded_imgs = self.encoder.predict(self.test_data)
        decoded_imgs = self.decoder.predict(encoded_imgs)



        n = 10  # how many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(1,n+1):
            # display original
            ax = plt.subplot(2, n, i)
            plt.plot(self.test_data[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + n)
            plt.plot(decoded_imgs[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

