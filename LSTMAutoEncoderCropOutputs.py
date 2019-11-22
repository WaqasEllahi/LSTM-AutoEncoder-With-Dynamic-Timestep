import os
import random

from keras import Input, Model
from keras.engine.saving import save_model, load_model
# from keras.layers import RepeatVector, K, LSTMAutoEncoderExample, Lambda, np
from keras.layers import RepeatVector,  LSTM, Lambda, Masking, TimeDistributed, Dense
from keras import backend as K
import numpy as np
from pyfolder import PyFolder



class LSTMAutoEncoderCropOutputs:
    """
    The LSTMAutoEncoderExample Autoencoder for dynamic timesteps series.
    This class can be used to train an LSTMAutoEncoderExample (no hidden layers yet) that behaves like an autoencoder for time series.
    It can be fed with unfixed timesteps series.
    """


    def __init__(self, latent_space, input_features, time_step, mask_val = 0.0):
        """
        Constructor of the autoencoder. Only latent space and the input features are required.
        Note that no timesteps are required to feed this LSTMAutoEncoderExample.
        :param latent_space: space to compress the data to.
        :param input_features: number of features that represent an element in the time serie.
        """

        self._latent_space = latent_space
        self._input_cells = input_features
        self._time_step = time_step
        self._mask_val = mask_val

        self._encoder = None
        self._decoder = None
        self._autoencoder = None
        self._configure_network()

    def _configure_network(self):
        """
        Sets up the network's layer.
        """

        def cropOutputs(x):
            # x[0] is decoded at the end
            # x[1] is inputs
            # both have the same shape

            # padding = 1 for actual data in inputs, 0 for 0
            padding = K.cast(K.not_equal(x[1], 0), dtype=K.floatx())
            # if you have zeros for non-padded data, they will lose their backpropagation

            return x[0] * padding

        encoder_input = Input(shape=(self._time_step, self._input_cells))
        masked_input = Masking(mask_value=self._mask_val, input_shape=(self._time_step, self._input_cells))(encoder_input)
        encoder_output = LSTM(self._latent_space, activation='relu', return_sequences=False)(masked_input)

        decoded_input = RepeatVector(self._time_step)(encoder_output)

        decoder_lstm_output = LSTM(self._latent_space, activation='relu', return_sequences=True)(decoded_input)
        tdOutput = TimeDistributed(Dense(self._input_cells))(decoder_lstm_output)
        decoder_output = Lambda(cropOutputs, output_shape=(self._time_step, self._input_cells))([tdOutput, encoder_input])

        self._autoencoder = Model(encoder_input, decoder_output)
        self._encoder = Model(encoder_input, encoder_output)

        self._autoencoder.compile(optimizer="Adam", loss="mse", metrics=["accuracy"])

    def encode(self, X):
        """
        Encodes the given input into a vector with the specified latent_space size.
        :param X: a numpy array of shape [1, N, input_features]
        :return: vector of shape [1, latent_space]
        """
        return self._encoder.predict(X)

    def predict(self, X):
        """
        Passes the specified element through the autoencoder and returns the result of the decoder.
        :param X: a numpy array of shape [1, N, input_features]
        :return: a numpy array of shape [1, N, input_features], that, if trained well, should be close to X.
        """
        return self._autoencoder.predict(X)

    def fit(self, X, epochs, ):
        """
        Fits the specified data into the autoencoder.
        :param X: a python's list containing elements, being each element a numpy array of shape [1, N, input_features].
                  Each element can contain a different "N" (timesteps).
        :param epochs: number of iterations through the whole X. On each iteration, X is shuffled.
        """
        for epoch in range(epochs):
            losses = []
            # We must select data from X. Like SGD.
            # The main issue is that each element of X might have different timesteps, that's the reason to select one
            # element randomly and apply the fit one at a time, computing epochs by ourselves
            random.shuffle(X)
            for element in X:
                loss = self._autoencoder.fit(element, element, epochs=1, verbose=0).history['loss'][0]
                losses.append(loss)

            print(f"Epoch loss: {np.mean(losses)}")


    #def evaluate(self, test_X):
    #    """
    #    Evaluates the autoencoder with the specified da
    #    :param test_X:
    #   :return:
    #    """
    #    return self._autoencoder.evaluate(test_X, test_X)

    def decode(self, X, timesteps):
        """
        Decodes the given vector into the corresponding time series.
        :param X: vector of shape [1, latent_space]
        :param timesteps: number of timesteps that the time-series originally had.
        :return: a numpy array of shape [1, timesteps, input_features]
        """
        return Decoder(self._autoencoder, self._latent_space).predict(X, timesteps)

    def save(self, uri):
        """
        Saves the model into a given filename.
        The model uses 4 files: one for the encoder, other for the decoder, other
        for the autoencoder and one for the class options in JSON format.
        :param uri: base filename.
        """
        pf = PyFolder(os.path.dirname(os.path.realpath(uri)), allow_override=True)
        pf[os.path.basename(uri)+"_options.json"] = {
            'input_cells': self._input_cells,
            'latent_space': self._latent_space,
        }

        save_model(self._autoencoder, uri+"_lstm_autoencoder.hdf5")
        save_model(self._encoder, uri+"_lstm_encoder.hdf5")

    def load(self, uri):
        """
        Loads the model from the specified URI.
        The model uses 4 files: one for the encoder, other for the decoder, other
        for the autoencoder and one for the class options in JSON format.
        :param uri: base filename
        """
        self._encoder = load_model(uri+"_lstm_encoder.hdf5")
        self._autoencoder = load_model(uri+"_lstm_autoencoder.hdf5")

        pf = PyFolder(os.path.dirname(os.path.realpath(uri)))
        dict_options = pf[os.path.basename(uri)+"_options.json"]

        self._latent_space = dict_options['latent_space']
        self._input_cells = dict_options['input_cells']

    @property
    def latent_space(self):
        return self._latent_space


class Decoder:
    """
    Decoder dynamic class.
    This class is required since the decoder must compute different output shape based on the number of timesteps
    desired, which are set dynamically when decoding in the autoencoder.
    """
    def __init__(self, autoencoder, latent_space):
        """
        Constructor of the Decoder.
        :param autoencoder: Keras model comprising the autoencoder.
        :param latent_space: number of elements in the compressed version of the data.
        """
        self._autoencoder = autoencoder
        self._latent_space = latent_space

    def predict(self, X, timesteps):
        """
        Decodes the given vector into the corresponding time series.
        :param X: vector of shape [1, latent_space]
        :param timesteps: number of timesteps that the time-series originally had.
        :return: a numpy array of shape [1, timesteps, input_features]
        """
        decoder_input = Input(shape=(self._latent_space,))

        decoder_repeated_input = RepeatVector(timesteps)(decoder_input)
        decoder = self._autoencoder.layers[-1](decoder_repeated_input)
        decoder = Model(decoder_input, decoder)

        return decoder.predict(X)