from abc import ABC

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class BaseDense(tf.keras.Sequential):

    def __init__(self, n_hidden=(10, 10), activation='relu'):
        self.n_hidden = n_hidden
        self.activation = activation
        layers = [self.dense_layer(units=units, activation=activation) for units in n_hidden]
        layers.append(self.dense_layer(units=1, activation=None))
        super().__init__(layers)

    def dense_layer(self, units, activation):
        raise NotImplementedError


class BaseBayesianDense(BaseDense, ABC):
    def __init__(self, n_samples, n_hidden=(10, 10), activation='relu'):
        self.n_samples = n_samples
        super().__init__(n_hidden=n_hidden, activation=activation)


class RegularDense(BaseDense):

    def dense_layer(self, units, activation):
        return tf.keras.layers.Dense(units=units, activation=activation) 


class FlipoutDense(BaseBayesianDense):

    def kl_divergence_function(self, q, p, _):
        return tfd.kl_divergence(q, p) / tf.cast(self.n_samples, dtype=tf.float32)

    def dense_layer(self, units, activation):
        return tfp.layers.DenseFlipout(units=units,
                                       kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                                       bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                                       kernel_divergence_fn=self.kl_divergence_function,
                                       activation=activation)
