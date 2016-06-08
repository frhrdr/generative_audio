from keras.layers import Input, Convolution1D, MaxPooling1D, UpSampling1D, ZeroPadding1D
from keras.models import Model
from keras.layers import BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K


class ConvAutoencoder:

    def __init__(self, train_set, test_set):
        fdim = train_set.shape[1]

        train_set = np.reshape(train_set, (train_set.shape[0], fdim, 1))
        test_set = np.reshape(test_set, (test_set.shape[0], fdim, 1))

        self.train_data = train_set
        self.test_data = test_set
        self.encoding_dim = train_set.shape[1]

        input_img = Input(shape=(fdim, 1))

        assert fdim % 8 == 0, "max pools require input divisible by 2^3"

        x = Convolution1D(32, 35, activation='relu', border_mode='same')(input_img)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2, border_mode='valid')(x)
        x = Convolution1D(16, 7, activation='relu', border_mode='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2, border_mode='valid')(x)
        x = Convolution1D(4, 3, activation='relu', border_mode='same')(x)
        x = BatchNormalization()(x)
        encoded = MaxPooling1D(2, border_mode='valid')(x)

        x = Convolution1D(4, 3, activation='relu', border_mode='same')(encoded)
        x = BatchNormalization()(x)
        x = UpSampling1D(2)(x)
        x = Convolution1D(16, 7, activation='relu', border_mode='same')(x)
        x = BatchNormalization()(x)
        x = UpSampling1D(2)(x)
        x = Convolution1D(32, 35, activation='relu', border_mode='same')(x)
        x = BatchNormalization()(x)
        x = UpSampling1D(2)(x)
        decoded = Convolution1D(1, 3, activation='tanh', border_mode='same')(x)

        self.autoencoder = Model(input=input_img, output=decoded)
        self.autoencoder.compile(optimizer='rmsprop', loss='mse')  # binary_crossentropy

        self.autoencoder.summary()

    def train(self, nb_epoch, batch_size, shuffle):

        self.autoencoder.fit(self.train_data, self.train_data,
                             nb_epoch=nb_epoch,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             validation_data=(self.train_data, self.train_data))

    def show(self):

        decoded_imgs = self.autoencoder.predict(self.test_data)

        num_datapoints = 20
        offset = 100
        n = 2
        plt.figure(figsize=(20, 4))
        for i in range(1, n+1):
            # display original
            ax = plt.subplot(2, n, i)
            plt.plot(self.test_data[i][offset:offset + num_datapoints])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + n)
            plt.plot(decoded_imgs[i][offset:offset + num_datapoints])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

