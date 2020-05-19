import numpy as np
import tensorflow_probability as tfp

from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow_core.python.keras.wrappers.scikit_learn import KerasClassifier

from active_learning_ratio_estimation.dataset import RatioDataset, SinglyParameterizedRatioDataset, \
    UnparameterizedRatioDataset, build_unparameterized_input, stack_repeat, build_singly_parameterized_input, ParamGrid

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

    def fit(self, dataset: RatioDataset):
        model_input = dataset.build_input()
        self.estimator.fit(model_input, dataset.y)
        return self

    def predict(self, *args):
        raise NotImplementedError

    def predict_proba(self, *args):
        raise NotImplementedError

    def predict_likelihood_ratio(self, x, *args):
        y_pred = self.predict_proba(x, *args)[:, 1]
        likelihood_ratio = y_pred / (1 - y_pred)
        return likelihood_ratio


class UnparameterizedRatioModel(BaseRatioModel):

    def fit(self, dataset: UnparameterizedRatioDataset):
        assert isinstance(dataset, UnparameterizedRatioDataset)
        super().fit(dataset)
        self.theta_0_ = dataset.theta_0
        self.theta_1_ = dataset.theta_1
        return self

    def predict(self, x):
        model_input = build_unparameterized_input(x)
        return self.estimator.predict(model_input)

    def predict_proba(self, x, *args):
        model_input = build_unparameterized_input(x)
        return self.estimator.predict_proba(model_input)


class SinglyParameterizedRatioModel(BaseRatioModel):

    def fit(self, dataset: SinglyParameterizedRatioDataset):
        assert isinstance(dataset, SinglyParameterizedRatioDataset)
        super().fit(dataset)
        self.theta_0_ = dataset.theta_0
        return self

    def predict(self, x, theta_1):
        assert len(theta_1) == len(x)
        model_input = build_singly_parameterized_input(x=x, theta_1s=theta_1)
        return self.estimator.predict(model_input)

    def predict_proba(self, x, theta_1):
        assert len(theta_1) == len(x)
        model_input = build_singly_parameterized_input(x=x, theta_1s=theta_1)
        return self.estimator.predict_proba(model_input)

    def nllr_param_scan(self, x: np.ndarray, param_grid: ParamGrid):
        # calculate the negative likelihood ratio across parameter grid for given x
        meshgrid = param_grid.meshgrid()



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
