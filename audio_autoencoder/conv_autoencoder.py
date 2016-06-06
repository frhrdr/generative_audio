from keras.layers import Input, Convolution1D, MaxPooling1D, UpSampling1D
from keras.models import Model
import matplotlib.pyplot as plt


class ConvAutoencoder:

    def __init__(self, train_set, test_set):

        #x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))




        self.train_data = train_set
        self.test_data = test_set
        self.encoding_dim = train_set.shape[1]

        input_img = Input(shape=(1000,))

        x = Convolution1D(16, 3, activation='relu', border_mode='same')(input_img)
        x = MaxPooling1D(2, border_mode='valid')(x)
        x = Convolution1D(8, 3, activation='relu', border_mode='same')(x)
        x = MaxPooling1D(2, border_mode='valid')(x)
    #    x = Convolution1D(8, 3, activation='relu', border_mode='same')(x)
    #    encoded = MaxPooling1D(2, border_mode='same')(x)

        #

    #    x = Convolution1D(8, 3, activation='relu', border_mode='same')(encoded)
    #    x = UpSampling1D(2)(x)
        x = Convolution1D(8, 3, activation='relu', border_mode='same')(x)
        x = UpSampling1D(2)(x)
        x = Convolution1D(16, 3, activation='relu')(x)
        x = UpSampling1D(2)(x)
        decoded = Convolution1D(1, 3, activation='tanh', border_mode='same')(x)

        self.autoencoder = Model(input=input_img, output=decoded)
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    def train(self, nb_epoch, batch_size, shuffle):


        self.autoencoder.fit(self.train_data, self.train_data,
                             nb_epoch=nb_epoch,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             validation_data=(self.test_data, self.test_data))

    def show(self):

        decoded_imgs = self.autoencoder.predict(self.test_data)

        n = 10
        plt.figure(figsize=(20, 4))
        for i in range(1, n+1):
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

