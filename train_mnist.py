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


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


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


def get_siamese(base_network, input_shape):
    def euclidean_distance(vects):
        x, y = vects
        return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

    def eucl_dist_output_shape(shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)

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

    base_network = get_base_network(input_shape=input_shape)
    siamese_network = get_siamese(base_network, input_shape)

    # train
    rms = RMSprop()
    siamese_network.compile(loss=contrastive_loss, optimizer=rms, metrics=['accuracy'])
    siamese_network.fit([x_pairs_train[:, 0], x_pairs_train[:, 1]], is_matched_train,
                        validation_data=([x_pairs_valid[:, 0], x_pairs_valid[:, 1]], is_matched_valid),
                        batch_size=128, epochs=100)

    siamese_network.save_weights("siamese.hdf5")

    siamese_network.load_weights("siamese.hdf5")

    def compute_accuracy(predictions, labels, thr):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        # pick labels predicted matched from labels
        labels_matched = labels[predictions.ravel() < thr]

        # pick labels predicted unmatched from labels
        labels_unmatched = labels[predictions.ravel() > thr]

        # create 0 / 1
        labels_predicted = np.array([1] * len(labels_matched) + [0] * len(labels_unmatched))

        # concatenate
        _labels = np.concatenate((labels_matched, labels_unmatched), axis=0)

        acc = len(_labels[_labels == labels_predicted]) / len(_labels)
        return acc

    # compute final accuracy on training and test sets
    pred = siamese_network.predict([x_pairs_train[:, 0], x_pairs_train[:, 1]])
    tr_acc = compute_accuracy(pred, is_matched_train, 0.5)
    pred = siamese_network.predict([x_pairs_valid[:, 0], x_pairs_valid[:, 1]])
    te_acc = compute_accuracy(pred, is_matched_valid, 0.5)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))


if __name__ == "__main__":
    main()
