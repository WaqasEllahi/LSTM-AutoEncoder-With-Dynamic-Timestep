from DataManipulation import Data
from LSTMAutoEncoderCropOutputs import LSTMAutoEncoderCropOutputs
import numpy as np
from LSTMAutoencoderWithRepeatVector import LSTMAutoencoderWithRepeatVector
from tensorflow.python.client import device_lib


def exampleWithDifferentTimestep():

    arr1 = np.array([[[0.3, 0.027],
                 [0.4, 0.064],
                 [0.5, 0.125]]])

    arr2 = np.array([[[0.4, 0.064],
                 [0.5, 0.125],
                 [0.6, 0.216],
                 [0.8, 0.512]]])

    arr3 = np.array([[[0.5, 0.125],
                  [0.6, 0.216],
                  [0.7, 0.343]]])

    arr4 = np.array([[[0.6, 0.216],
                 [0.7, 0.343],
                 [0.8, 0.512],
                 [0.6, 0.216]]])

    arr5 = np.array([[[0.7, 0.343],
                 [0.8, 0.512],
                 [0.9, 0.729]]])

    inputs = [arr1, arr2, arr3, arr4, arr5]

    autoencoder = LSTMAutoencoderWithRepeatVector(latent_space=64, input_features=2)
    autoencoder.fit(inputs, epochs=100)

    for X in inputs:
        x_hat = autoencoder.predict(X)
        print('---Predicted---')
        print(np.round(x_hat, 3))
        print('---Actual---')
        print(np.round(X, 3))



def exampleForDifferentTimestepWithMasking():

    arr1 = np.array([[[0.3, 0.027],
                      [0.4, 0.064],
                      [0.5, 0.125],
                      [0.0, 0.0]]])

    arr2 = np.array([[[0.4, 0.064],
                      [0.5, 0.125],
                      [0.6, 0.216],
                      [0.8, 0.512]]])

    arr3 = np.array([[[0.5, 0.125],
                      [0.6, 0.216],
                      [0.7, 0.343],
                      [0.0, 0.0]]])

    arr4 = np.array([[[0.6, 0.216],
                      [0.7, 0.343],
                      [0.8, 0.512],
                      [0.6, 0.216]]])

    arr5 = np.array([[[0.7, 0.343],
                      [0.8, 0.512],
                      [0.9, 0.729],
                      [0.0, 0.0]]])

    inputs = [arr1, arr2, arr3, arr4, arr5]

    autoencoder = LSTMAutoEncoderCropOutputs(latent_space=64, input_features=2, time_step=4, mask_val=0.0)
    autoencoder.fit(inputs, epochs=100)

    for X in inputs:
        x_hat = autoencoder.predict(X)
        print('---Predicted---')
        print(np.round(x_hat, 3))
        print('---Actual---')
        print(np.round(X, 3))


def main():

    exampleWithDifferentTimestep()
    # exampleForDifferentTimestepWithMasking()



if __name__== "__main__":
    main()