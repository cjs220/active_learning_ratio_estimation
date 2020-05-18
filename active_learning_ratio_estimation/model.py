import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from sklearn.calibration import CalibratedClassifierCV
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.wrappers.scikit_learn import KerasClassifier

from active_learning_ratio_estimation.dataset import RatioDataset, Dataset

tfd = tfp.distributions


def build_input_unparameterised(x, theta_0, theta_1):
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    return x


def build_input_single_parameterised(x, theta_0, theta_1):
    assert len(np.unique(theta_0) == theta_0.shape[1])
    return np.concatenate([x, theta_1], axis=1)


def build_input_double_parameterised(x, theta_0, theta_1):
    return np.concatenate([x, theta_0, theta_1], axis=1)


def sequential_build_fn(dense_layer,
                        n_hidden=(10, 10),
                        activation='relu',
                        optimizer='adam',
                        loss='bce',
                        metrics=None):
    layers = [dense_layer(units, activation=activation) for units in n_hidden]
    layers.append(dense_layer(1, 'sigmoid'))
    model = Sequential(layers=layers)
    model.compile(metrics=metrics, loss=loss, optimizer=optimizer)
    return model


def regular_sequential_build_fn(self,
                                n_samples=None,
                                n_hidden=(10, 10),
                                activation='relu',
                                optimizer='adam',
                                loss='bce',
                                metrics=None,
                                callbacks=None):

    def dense_layer(units, activation):
        return tf.keras.layers.Dense(units=units, activation=activation)

    return sequential_build_fn(dense_layer=dense_layer,
                               n_hidden=n_hidden,
                               activation=activation,
                               optimizer=optimizer,
                               loss=loss,
                               metrics=metrics)


def bayesian_sequential_build_fn(self,
                                 n_samples,
                                 n_hidden=(10, 10),
                                 activation='relu',
                                 optimizer='adam',
                                 loss='bce',
                                 metrics=None,
                                 callbacks=None):
    def kl_divergence_function(q, p, _):
        return tfd.kl_divergence(q, p) / tf.cast(n_samples, dtype=tf.float32)

    def dense_layer(units, activation):
        return tfp.layers.DenseFlipout(units=units,
                                       kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                                       bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                                       kernel_divergence_fn=kl_divergence_function,
                                       activation=activation)

    return sequential_build_fn(dense_layer=dense_layer,
                               n_hidden=n_hidden,
                               activation=activation,
                               optimizer=optimizer,
                               loss=loss,
                               metrics=metrics)


class RatioModel:
    def __init__(self,
                 parameterisation,
                 n_samples,
                 n_hidden=(10, 10),
                 activation='relu',
                 optimizer='adam',
                 loss='bce',
                 metrics=None,
                 calibration_method=None,
                 cv=1,
                 fit_kwargs=None
                 ):
        self.parameterisation = parameterisation
        self.build_input = {
            0: build_input_unparameterised,
            1: build_input_single_parameterised,
            2: build_input_double_parameterised
        }[parameterisation]
        callbacks = fit_kwargs.pop('callbacks', None)
        self.estimator = KerasClassifier(build_fn=self.build_fn,
                                         n_samples=n_samples,
                                         n_hidden=n_hidden,
                                         activation=activation,
                                         optimizer=optimizer,
                                         loss=loss,
                                         metrics=metrics,
                                         callbacks=callbacks,
                                         **fit_kwargs)

        self.calibration_method = calibration_method
        if calibration_method is not None:
            self.estimator = CalibratedClassifierCV(base_estimator=self.estimator,
                                                    method=self.calibration_method,
                                                    cv=cv)

    @property
    def build_fn(self):
        raise NotImplementedError

    def fit(self, x, theta_0, theta_1, y):
        model_input = self.build_input(x, theta_0, theta_1)
        self.estimator.fit(model_input, y)
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
        return self.predict_likelihood_ratio(dataset.x, dataset.theta_0, dataset.theta_1)


class RegularRatioModel(RatioModel):
    build_fn = regular_sequential_build_fn


class BayesianRatioModel(RatioModel):
    build_fn = bayesian_sequential_build_fn
    prediction_mc_samples = 100

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


def param_scan_single(model: RatioModel,
                      X: np.array,
                      theta_1s: np.array,
                      theta_0: np.array):
    assert model.parameterisation == 1
    assert len(theta_0.shape) == 1
    ds = Dataset()

    def _tile_reshape(theta):
        return np.tile(theta, len(X)).reshape(len(X), len(theta))

    _theta_0 = _tile_reshape(theta_0)

    for theta_1 in map(_tile_reshape, theta_1s):
        ds = ds + Dataset.from_arrays(x=X, theta_0=_theta_0, theta_1=theta_1)

    lr_pred = model.predict_likelihood_ratio_dataset(ds)
    collected = np.stack(np.split(lr_pred, len(theta_1s)))
    return -2 * np.log(collected).sum(axis=1)
