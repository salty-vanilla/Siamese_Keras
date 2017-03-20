from keras.layers import (
    Input,
    Lambda
)
from keras import backend as K
from keras.models import Model
import os
import numpy as np


class Siamese:
    def __init__(self, input_shape, base_network, param_path_base_network=None):
        self.base_network = base_network
        self.input_shape = input_shape
        self.siamese_network = None
        self.param_path_base_network = param_path_base_network
        self.trained = False

        if param_path_base_network and os.path.exists(param_path_base_network):
            self.trained = True
        else:
            self.param_path_base_network = None

    def build(self, optimizer):
        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)

        # because we re-use the same instance `base_network`,
        # the weights of the network
        # will be shared across the two branches
        processed_a = self.base_network(input_a)
        processed_b = self.base_network(input_b)

        distance = Lambda(self.euclidean_distance,
                          output_shape=self.eucl_dist_output_shape)([processed_a, processed_b])

        self.siamese_network = Model([input_a, input_b], distance)
        self.siamese_network.compile(loss=self.contrastive_loss,
                                     optimizer=optimizer)

    def fit(self, x_pairs_train, is_matched_train,
            x_pairs_valid=None, is_matched_valid=None,
            batch_size=32, epochs=100,
            param_path="./params/siamese.hdf5", callbacks=None):

        if x_pairs_valid and is_matched_valid:
            self.siamese_network.fit([x_pairs_train[:, 0], x_pairs_train[:, 1]], is_matched_train,
                                     validation_data=([x_pairs_valid[:, 0], x_pairs_valid[:, 1]], is_matched_valid),
                                     batch_size=batch_size, epochs=epochs, callbacks=callbacks)
        else:
            self.siamese_network.fit([x_pairs_train[:, 0], x_pairs_train[:, 1]], is_matched_train,
                                     batch_size=batch_size, epochs=epochs, callbacks=callbacks)

        os.makedirs(os.path.split(param_path)[0], exist_ok=True)
        self.siamese_network.save_weights(param_path)

        self.trained = True

    def predict(self, x_pairs):
        if self.trained and not self.param_path_base_network:
            pass
        else:
            self.base_network.load_weights(self.param_path_base_network)

        predictions = self.siamese_network.predict([x_pairs[:, 0], x_pairs[:, 1]])
        return predictions

    def eval(self, x_pairs, is_matched, thr=0.5):
        predictions = self.predict(x_pairs)
        acc = self.compute_accuracy(predictions, is_matched, thr)
        return acc

    @staticmethod
    def euclidean_distance(vects):
        x, y = vects
        return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

    @staticmethod
    def eucl_dist_output_shape(shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    @staticmethod
    def contrastive_loss(y_true, y_pred):
        margin = 1
        return K.mean(y_true * K.square(y_pred) +
                      (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

    @staticmethod
    def compute_accuracy(predictions, labels, thr):
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
