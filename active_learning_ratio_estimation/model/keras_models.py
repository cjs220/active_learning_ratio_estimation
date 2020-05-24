from collections import Sequence

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from active_learning_ratio_estimation.model.ratio_model import tfd


class _BaseFeedForward(tf.keras.Model):
    def __init__(self, n_hidden=(10, 10), activation='relu', dropout=None):
        super().__init__()
        self.n_hidden = n_hidden
        self.activation = activation
        self.dense_layers = self.build_dense_layers()
        self.output_layer = self.dense_layer(1, activation=None)

    def call(self, inputs, training=False, **kwargs):
        for layer_name, layer in self.dense_layers:
            inputs = layer(inputs)
        logits = self.output_layer(inputs)
        probs = tf.sigmoid(logits)
        return probs

    def predict_proba(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def build_dense_layers(self):
        dense_layers = []
        for i in range(len(self.n_hidden)):
            dense_layer = self.dense_layer(units=self.n_hidden[i], activation=self.activation)
            layer_name = f'Dense {i + 1}'
            dense_layers.append((layer_name, dense_layer))
            return dense_layers

    def dense_layer(self, units, activation, regularizer=None):
        raise NotImplementedError


class BaseFeedForward(tf.keras.Sequential):

    def __init__(self, n_hidden=(10, 10), activation='relu', dropout=0.0, l1_regularization=0.0,
                 l2_regularization=0.0):
        self.n_hidden = n_hidden
        self.activation = activation
        self.dropout = dropout
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization
        super().__init__(self.build_layers())

    def build_layers(self):
        l1s = self._resolve_sequence_arg(self.l1_regularization, len(self.n_hidden))
        l2s = self._resolve_sequence_arg(self.l2_regularization, len(self.n_hidden))
        dropouts = self._resolve_sequence_arg(self.dropout, len(self.n_hidden))
        regularizers = [tf.keras.regularizers.L1L2(l1=l1, l2=l2) for l1, l2 in zip(l1s, l2s)]
        layers = []
        for i in range(len(self.n_hidden)):
            layers.append(self.dense_layer(units=self.n_hidden[i],
                                           activation=self.activation,
                                           regularizer=regularizers[i]))
            layers.append(tf.keras.layers.Dropout(dropouts[i]))
        layers.append(self.dense_layer(1, 'sigmoid'))
        return layers

    def dense_layer(self, units, activation, regularizer=None):
        raise NotImplementedError

    @staticmethod
    def _resolve_sequence_arg(possible_seq, length):
        if not isinstance(possible_seq, Sequence):
            seq = [possible_seq] * length
        else:
            seq = possible_seq
        return seq


class BaseBayesianFeedForward(BaseFeedForward):
    prediction_mc_samples = 100

    def __init__(self, n_samples, n_hidden=(10, 10), activation='relu'):
        self.n_samples = n_samples
        super().__init__(n_hidden=n_hidden, activation=activation)

    def dense_layer(self, units, activation, regularizer=None):
        raise NotImplementedError

    def predict_proba(self, x, **kwargs):
        x_tile = np.repeat(x, self.prediction_mc_samples, axis=0)
        preds = super(BaseBayesianFeedForward, self).predict_proba(x_tile, **kwargs).squeeze()
        stack_preds = np.stack(np.split(preds, len(x)))
        y_pred = stack_preds.mean(axis=1)
        return y_pred.reshape(-1, 1)


class FeedForward(_BaseFeedForward):

    def dense_layer(self, units, activation, regularizer=None):
        return tf.keras.layers.Dense(units=units, activation=activation,
                                     bias_regularizer=regularizer, kernel_regularizer=regularizer)


class FlipoutFeedForward(BaseBayesianFeedForward):

    def kl_divergence_function(self, q, p, _):
        return tfd.kl_divergence(q, p) / tf.cast(self.n_samples, dtype=tf.float32)

    def dense_layer(self, units, activation, regularizer=None):
        return tfp.layers.DenseFlipout(units=units,
                                       kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                                       bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                                       kernel_divergence_fn=self.kl_divergence_function,
                                       activation=activation)


def build_feedforward(n_hidden=(10, 10),
                      activation='relu',
                      optimizer='adam',
                      loss='bce',
                      metrics=None,
                      callbacks=None,
                      dropout=0.0):
    model = FeedForward(n_hidden=n_hidden, activation=activation, dropout=dropout)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=False)
    return model


def build_bayesian_flipout(n_samples,
                           n_hidden=(10, 10),
                           activation='relu',
                           optimizer='adam',
                           loss='bce',
                           metrics=None,
                           callbacks=None):
    model = FlipoutFeedForward(n_samples=n_samples, n_hidden=n_hidden, activation=activation)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
