from abc import ABC

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class BaseFeedForward(tf.keras.Model):
    def __init__(self, n_hidden=(10, 10), activation='relu'):
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

    def dense_layer(self, units, activation):
        raise NotImplementedError


class BaseBayesianFeedForward(BaseFeedForward, ABC):
    def __init__(self, n_samples, n_hidden=(10, 10), activation='relu'):
        self.n_samples = n_samples
        super().__init__(n_hidden=n_hidden, activation=activation)


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
