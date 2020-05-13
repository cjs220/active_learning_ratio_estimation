import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.calibration import CalibratedClassifierCV
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.wrappers.scikit_learn import KerasClassifier

from dataset import RatioDataset

tfd = tfp.distributions


class RatioModel:

    def __init__(self,
                 x_dim,
                 theta_dim,
                 num_samples,
                 n_hidden=(10, 10),
                 activation='relu',
                 output_activation='sigmoid',
                 calibration=None,
                 compile_kwargs=None,
                 fit_kwargs=None):
        self.x_dim = x_dim
        self.theta_dim = theta_dim
        self.num_samples = num_samples
        self.n_hidden = n_hidden
        self.activation = activation
        self.output_activation = output_activation
        self.calibration = calibration
        if calibration is not None:
            assert calibration in ('isotonic', 'sigmoid')
        self.compile_kwargs = compile_kwargs if compile_kwargs is not None else dict()
        self.fit_kwargs = fit_kwargs if fit_kwargs is not None else dict()
        self.fit_kwargs['callbacks'] = self.compile_kwargs.pop('callbacks', None)
        self.estimator = KerasClassifier(build_fn=self._build_fn, **fit_kwargs)

    def _build_fn(self):
        model = Sequential()
        dense_layers = self._build_layers(n_hidden=self.n_hidden,
                                          activation=self.activation,
                                          output_activation=self.output_activation)
        for layer in dense_layers:
            model.add(layer)
        model.compile(**self.compile_kwargs)
        return model

    def _build_layers(self, n_hidden=(10, 10), activation='relu', output_activation='sigmoid'):
        layers = [self.dense_layer(units, activation) for units in n_hidden]
        layers.append(self.dense_layer(1, output_activation))
        return layers

    @property
    def calibrated(self):
        return self.calibration is not None

    def fit(self, x, theta_0, theta_1, y):
        model_input = self.build_input(x, theta_0, theta_1)
        self.estimator.fit(model_input, y)

        if self.calibrated:
            self.estimator = CalibratedClassifierCV(
                base_estimator=self.estimator,
                cv='prefit',
                method=self.calibration
            ).fit(model_input, y)

        return self

    def fit_dataset(self, dataset: RatioDataset):
        self.fit(dataset.x, dataset.theta_0, dataset.theta_1, dataset.y)
        return self

    def predict(self, x, theta_0, theta_1):
        model_input = self.build_input(x, theta_0, theta_1)
        return self.estimator.predict(model_input)

    def predict_proba(self, x, theta_0, theta_1):
        model_input = self.build_input(x, theta_0, theta_1)
        return self.estimator.predict_proba(model_input)

    def predict_likelihood_ratio(self, x, theta_0, theta_1):
        y_pred = self.predict_proba(x, theta_0, theta_1)[:, 1]
        likelihood_ratio = y_pred / (1 - y_pred)
        return likelihood_ratio

    def predict_dataset(self, dataset):
        return self.predict(dataset.x, dataset.theta_0, dataset.theta_1)

    def predict_proba_dataset(self, dataset):
        return self.predict_proba(dataset.x, dataset.theta_0, dataset.theta_1)

    def predict_likelihood_ratio_dataset(self, dataset):
        return self.predict_likelihood_ratio(dataset.x, dataset.theta, dataset.theta_1)

    @property
    def input_dim(self):
        raise NotImplementedError

    def build_input(self, x, theta_1, theta_0):
        raise NotImplementedError


class UnparameterisedRatioModel(RatioModel):

    @property
    def input_dim(self):
        return self.x_dim

    def build_input(self, x, theta_1, theta_0):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        return x


class SingleParameterisedRatioModel(RatioModel):

    @property
    def input_dim(self):
        return self.x_dim + self.theta_dim

    def build_input(self, x, theta_1, theta_0):
        # TODO
        pass


class DoubleParameterisedRatioModel(RatioModel):

    @property
    def input_dim(self):
        return self.x_dim + 2 * self.theta_dim

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
