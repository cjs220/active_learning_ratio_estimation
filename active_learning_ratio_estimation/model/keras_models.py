from abc import ABC
from typing import Sequence

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class BaseDense(tf.keras.Sequential):

    def __init__(
            self,
            activation: str,
            n_hidden: Sequence[int] = (10, 10)
    ):
        self.n_hidden = n_hidden
        self.activation = activation
        layers = [self._dense_layer(units=units, activation=activation) for units in n_hidden]
        layers.append(self._dense_layer(units=1, activation=None))
        super().__init__(layers)

    def _dense_layer(self, units, activation):
        raise NotImplementedError


class BaseBayesianDense(BaseDense, ABC):

    def __init__(
            self,
            n_samples,
            activation: str = 'relu',
            n_hidden: Sequence[int] = (10, 10),
    ):
        self.n_samples = n_samples
        super().__init__(n_hidden=n_hidden, activation=activation)


class RegularDense(BaseDense):

    def __init__(
            self,
            activation: str = 'tanh',
            n_hidden: Sequence[int] = (10, 10),
            l2_regularization: float = 0.0,
    ):
        self.l2_regularization = l2_regularization
        super().__init__(activation=activation, n_hidden=n_hidden)

    def _regularizer(self):
        return tf.keras.regularizers.l2(l=self.l2_regularization)

    def _dense_layer(self, units, activation):
        return tf.keras.layers.Dense(
            units=units,
            activation=activation,
            kernel_regularizer=self._regularizer()
        )


class FlipoutDense(BaseBayesianDense):

    def _kl_divergence_function(self, q, p, _):
        return tfd.kl_divergence(q, p) / tf.cast(self.n_samples, dtype=tf.float32)

    def _dense_layer(self, units, activation):
        return tfp.layers.DenseFlipout(
            units=units,
            kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
            bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
            kernel_divergence_fn=self._kl_divergence_function,
            activation=activation
        )
