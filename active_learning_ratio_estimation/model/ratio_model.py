import numpy as np
import tensorflow_probability as tfp

from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow_core.python.keras.wrappers.scikit_learn import KerasClassifier

from active_learning_ratio_estimation.dataset import RatioDataset

tfd = tfp.distributions


class BaseRatioModel:
    def __init__(self,
                 build_fn,
                 fit_kwargs=None,
                 build_fn_kwargs=None,
                 calibration_method=None,
                 cv=1,
                 normalize_input=False):
        build_fn_kwargs['callbacks'] = fit_kwargs.pop('callbacks', None)
        clf = KerasClassifier(build_fn, **build_fn_kwargs, **fit_kwargs)
        steps = [('clf', clf)]
        if normalize_input:
            steps.insert(0, ('StandardScaler', StandardScaler()))
        self.estimator = Pipeline(steps=steps)

        self.calibration_method = calibration_method
        if calibration_method is not None:
            self.estimator = CalibratedClassifierCV(base_estimator=self.estimator,
                                                    method=self.calibration_method,
                                                    cv=cv)

    def build_input(self, x, theta_0, theta_1):
        raise NotImplementedError

    def fit(self, x, theta_0, theta_1, y):
        model_input = self.build_input(x, theta_0, theta_1)
        self.estimator.fit(model_input, y)
        return self

    def fit_dataset(self, dataset: RatioDataset):
        self.fit(dataset.x, dataset.theta_0s, dataset.theta_1s, dataset.y)
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
        return self.predict(dataset.x, dataset.theta_0s, dataset.theta_1s)

    def predict_proba_dataset(self, dataset):
        return self.predict_proba(dataset.x, dataset.theta_0s, dataset.theta_1s)

    def predict_likelihood_ratio_dataset(self, dataset):
        return self.predict_likelihood_ratio(dataset.x, dataset.theta_0s, dataset.theta_1s)


class UnparameterizedRatioModel(BaseRatioModel):

    def build_input(self, x, theta_0, theta_1):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        return x


class SinglyParameterizedRatioModel(BaseRatioModel):

    def build_input(self, x, theta_0, theta_1):
        assert len(np.unique(theta_0) == theta_0.shape[1])
        return np.concatenate([x, theta_1], axis=1)


class DoublyParameterizedRatioModel(BaseRatioModel):

    def build_input(self, x, theta_0, theta_1):
        return np.concatenate([x, theta_0, theta_1], axis=1)


def param_scan_single(model: BaseRatioModel,
                      X: np.array,
                      theta_1s: np.array,
                      theta_0: np.array):
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
