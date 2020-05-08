import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from dataset import RatioDataset

tfd = tfp.distributions


class RatioModel(tf.keras.Model):

    def __init__(self,
                 input_dim,
                 num_samples,
                 n_hidden=(10, 10),
                 activation='relu',
                 output_activation='sigmoid'):
        super().__init__()
        self.input_dim = input_dim
        self.num_samples = num_samples
        self.dense_layers = self.build_layers(n_hidden=n_hidden,
                                              activation=activation,
                                              output_activation=output_activation)

    def __call__(self, inputs, training=False, **kwargs):
        for layer in self.dense_layers:
            inputs = layer(inputs)
        return inputs

    def fit_dataset(self, dataset: RatioDataset, **kwargs):
        model_input = self.build_input(dataset.x, dataset.theta_1, dataset.theta_0)
        self.fit(x=model_input, y=dataset.y, **kwargs)

    def predict(self, x, theta_0, theta_1, **kwargs):
        model_input = self.build_input(x, theta_0, theta_1)
        return super().predict(model_input, **kwargs)

    def predict_dataset(self, ds, **kwargs):
        return self.predict(ds.x, ds.theta_0, ds.theta_1, **kwargs)

    def predict_likelihood_ratio(self, x, theta_0, theta_1, **kwargs):
        y_pred = self.predict(x, theta_0, theta_1, **kwargs)
        likelihood_ratio = y_pred / (1 - y_pred)
        return likelihood_ratio

    def predict_likelihood_ratio_dataset(self, ds, **kwargs):
        return self.predict_likelihood_ratio(ds.x, ds.theta, ds.theta_1, **kwargs)

    def build_input(self, x, theta_1, theta_0):
        raise NotImplementedError

    def build_layers(self, n_hidden=(10, 10), activation='relu', output_activation='sigmoid'):
        layers = [self.dense_layer(units, activation) for units in n_hidden]
        layers.append(self.dense_layer(1, output_activation))
        return layers


class UnparameterisedRatioModel(RatioModel):

    def __init__(self, x_dim, num_samples, n_hidden=(10, 10), activation='relu', output_activation='sigmoid'):
        input_dim = x_dim
        super().__init__(input_dim=input_dim,
                         num_samples=num_samples,
                         n_hidden=n_hidden,
                         activation=activation,
                         output_activation=output_activation)

    def build_input(self, x, theta_1, theta_0):
        return x


class SingleParameterisedRatioModel(RatioModel):

    def __init__(self,
                 x_dim,
                 theta_dim,
                 num_samples,
                 n_hidden=(10, 10),
                 activation='relu',
                 output_activation='sigmoid'):
        input_dim = x_dim + theta_dim
        super().__init__(input_dim=input_dim,
                         num_samples=num_samples,
                         n_hidden=n_hidden,
                         activation=activation,
                         output_activation=output_activation)

    def build_input(self, x, theta_1, theta_0):
        # TODO
        pass


class DoubleParameterisedRatioModel(RatioModel):

    def __init__(self,
                 x_dim,
                 theta_dim,
                 num_samples,
                 n_hidden=(10, 10),
                 activation='relu',
                 output_activation='sigmoid'):
        input_dim = x_dim + 2 * theta_dim
        super().__init__(input_dim=input_dim,
                         num_samples=num_samples,
                         n_hidden=n_hidden,
                         activation=activation,
                         output_activation=output_activation)

    def build_input(self, x, theta_1, theta_0):
        # TODO
        pass


class FrequentistMixin:

    def dense_layer(self, units, activation):
        return tf.keras.layers.Dense(units=units, activation=activation)


class BayesianMixin:
    num_samples = None

    def kl_divergence_function(self, q, p, _):
        return tfd.kl_divergence(q, p) / tf.cast(self.num_samples, dtype=tf.float32)

    def dense_layer(self, units, activation):
        return tfp.layers.DenseFlipout(units=units,
                                       kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                                       bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                                       kernel_divergence_fn=self.kl_divergence_function,
                                       activation=activation)


class FrequentistUnparameterisedRatioModel(UnparameterisedRatioModel, FrequentistMixin):
    pass


class BayesianUnparameterisedRatioModel(UnparameterisedRatioModel, BayesianMixin):
    pass
