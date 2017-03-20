# -*- coding: utf-8 -*-
from __future__ import print_function
import os
try:
    import cPickle as pickle
except:
    import pickle
import gzip
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import numpy as np
from interface import Siamese
from keras.layers import (
    Input,
    Conv2D,
    MaxPool2D,
    Lambda,
    Flatten,
    Dense
)
from keras.models import Sequential, Model
from keras import backend as K
from keras.optimizers import RMSprop


width = height = 28
channel = 1


def load_mnist():
    print("Loading MNIST ...    ", end="")
    f = gzip.open('mnist.pkl.gz', 'rb')
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = pickle.load(f, encoding='latin1')

    if K.image_dim_ordering() == 'th':
        x_train = x_train.reshape((x_train.shape[0], channel, height, width))
        x_valid = x_valid[0].reshape((x_valid.shape[0], channel, height, width))
        x_test = x_test[0].reshape((x_test.shape[0], channel, height, width))
    else:
        x_train = x_train.reshape((x_train.shape[0], height, width, channel))
        x_valid = x_valid.reshape((x_valid.shape[0], height, width, channel))
        x_test = x_test.reshape((x_test.shape[0], height, width, channel))

    x_train = x_train.astype('float32') / 255
    x_valid = x_valid.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    print("COMPLETE")
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def create_pairs(x, digit_indexes):
    pairs = []
    labels = []
    n = min([len(digit_indexes[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            # same digits
            z1, z2 = digit_indexes[d][i], digit_indexes[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            
            # difference digits
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indexes[d][i], digit_indexes[dn][i]
            pairs += [[x[z1], x[z2]]]
            
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def get_base_network(input_shape):
    print("Building Generator ...   ", end="")

    model = Sequential(name="base_network")
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    print("COMPLETE")
    return model


def main():
    if K.image_dim_ordering() == 'th':
        input_shape = (channel, height, width)
    else:
        input_shape = (height, width, channel)

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_mnist()

    # create training+test positive and negative pairs
    digit_indexes = [np.where(y_train == i)[0] for i in range(10)]
    x_pairs_train, is_matched_train = create_pairs(x_train, digit_indexes)

    digit_indexes = [np.where(y_valid == i)[0] for i in range(10)]
    x_pairs_valid, is_matched_valid = create_pairs(x_valid, digit_indexes)

    digit_indexes = [np.where(y_test == i)[0] for i in range(10)]
    x_pairs_test, is_matched_test = create_pairs(x_test, digit_indexes)

    base_network = get_base_network(input_shape=input_shape)
    siamese = Siamese(input_shape=input_shape,
                      base_network=base_network)
    rms = RMSprop()
    siamese.compile(rms)

    # train
    siamese.network.fit([x_pairs_train[:, 0], x_pairs_train[:, 1]], is_matched_train,
                        validation_data=([x_pairs_valid[:, 0], x_pairs_valid[:, 1]], is_matched_valid),
                        batch_size=300, epochs=5)

    siamese.network.save_weights("siamese.hdf5")

    acc_test = siamese.eval(x_pairs_test, is_matched_test)

    print('* Accuracy on test set: %0.2f%%' % (100 * acc_test))


if __name__ == "__main__":
    main()
