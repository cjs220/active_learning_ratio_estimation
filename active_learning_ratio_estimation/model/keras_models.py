import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from active_learning_ratio_estimation.model.ratio_model import tfd


class BaseFeedForward(tf.keras.Sequential):

    def __init__(self, n_hidden=(10, 10), activation='relu'):
        self.n_hidden = n_hidden
        self.activation = activation
        layers = [self.dense_layer(units, activation=activation) for units in n_hidden]
        layers.append(self.dense_layer(1, 'sigmoid'))
        super().__init__(layers)

    def dense_layer(self, units, activation):
        raise NotImplementedError


class BaseBayesianFeedForward(BaseFeedForward):
    prediction_mc_samples = 100

    def __init__(self, n_samples, n_hidden=(10, 10), activation='relu'):
        self.n_samples = n_samples
        super().__init__(n_hidden=n_hidden, activation=activation)

    def dense_layer(self, units, activation):
        raise NotImplementedError

    def predict(self, x, theta_0, theta_1, n_samples=None):
        proba_preds = self.predict_proba(x, theta_0, theta_1, n_samples=n_samples)[:, 1]
        return np.around(proba_preds)

    def predict_proba(self, x, theta_0, theta_1, n_samples=None):
        n_samples = n_samples or self.prediction_mc_samples
        x_tile = np.tile(x, n_samples)
        preds = super(self.__class__, self).predict_proba(x_tile, theta_0, theta_1)
        stack_preds = np.stack(np.split(preds, n_samples))
        y_pred = stack_preds.mean(axis=0)
        return y_pred


class FeedForward(BaseFeedForward):

    def dense_layer(self, units, activation):
        return tf.keras.layers.Dense(units=units, activation=activation)


class FlipoutFeedForward(BaseBayesianFeedForward):

    def kl_divergence_function(self, q, p, _):
        return tfd.kl_divergence(q, p) / tf.cast(self.n_samples, dtype=tf.float32)

    def dense_layer(self, units, activation):
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
                      callbacks=None):
    model = FeedForward(n_hidden=n_hidden, activation=activation)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
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