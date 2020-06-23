from typing import Dict, Sequence, Callable

import numpy as np
from tqdm import tqdm
import tensorflow as tf

from active_learning_ratio_estimation.dataset import ParamGrid, build_singly_parameterized_input, \
    SinglyParameterizedRatioDataset, SingleParamIterator
from active_learning_ratio_estimation.util import estimated_likelihood_ratio, estimated_log_likelihood_ratio, \
    stack_repeat, outer_prod_shape_to_meshgrid_shape, build_simulator
from carl.learning import CalibratedClassifierCV


class BaseRatioModel:
    def __init__(self, clf):
        self.clf = clf

    def _fit(self, X: np.ndarray, y: np.ndarray, **fit_params):
        self.clf.fit(X, y, **fit_params)  # TODO
        return self

    def _predict(self, X: np.ndarray, log=False, return_std=False, **predict_params):
        if log:
            if hasattr(self.clf, 'predict_logits'):
                prediction = self._predict_log_from_logits(X=X, return_std=return_std, **predict_params)
            else:
                assert not return_std
                y_prob = self.clf.predict_proba(X=X, **predict_params)[:, 1]
                prediction = estimated_log_likelihood_ratio(y_prob)
        else:
            assert not return_std
            y_prob = self.clf.predict_proba(X=X, **predict_params)[:, 1]
            prediction = estimated_likelihood_ratio(y_prob)
        return prediction

    def _predict_log_from_logits(self, X: np.ndarray, return_std: bool, **predict_params):
        if return_std:
            predict_params['return_std'] = True

        return self.clf.predict_logits(X=X, **predict_params)


class UnparameterizedRatioModel(BaseRatioModel):

    def __init__(self, theta_0, theta_1, clf):
        self._theta_0 = theta_0
        self._theta_1 = theta_1
        super().__init__(clf)

    @property
    def theta_0(self):
        return self._theta_0

    @property
    def theta_1(self):
        return self._theta_1

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params):
        return self._fit(X, y, **fit_params)

    def predict(self, X: np.ndarray, log=False, **predict_params):
        return self._predict(X, log=log, **predict_params)


class SinglyParameterizedRatioModel(BaseRatioModel):

    def __init__(self, theta_0, clf):
        self._theta_0 = theta_0
        super().__init__(clf)

    @property
    def theta_0(self):
        return self._theta_0

    def fit(self, X: np.ndarray, theta_1s: np.ndarray, y: np.ndarray, **fit_params):
        model_input = build_singly_parameterized_input(X, theta_1s)
        return self._fit(model_input, y, **fit_params)

    def predict(self, X: np.ndarray, theta_1s: np.ndarray, log=False, **predict_params):
        model_input = build_singly_parameterized_input(X, theta_1s)
        return self._predict(model_input, log=log, **predict_params)

    def calibrated_predict(self,
                           X: np.ndarray,
                           theta: np.ndarray,
                           n_samples_per_theta: int,
                           simulator_func: Callable,
                           calibration_params: Dict,
                           log=False,
                           return_calibrated_model=False,
                           ):
        cal_clf = CalibratedClassifierCV(base_estimator=self.clf, cv='prefit', **calibration_params)
        cal_model = self.__class__(theta_0=self.theta_0, clf=cal_clf)
        calibration_ds = SinglyParameterizedRatioDataset.from_simulator(
            simulator_func=simulator_func,
            theta_0=self.theta_0,
            theta_1_iterator=SingleParamIterator(theta),
            n_samples_per_theta=n_samples_per_theta
        )
        cal_model.fit(X=calibration_ds.x, theta_1s=calibration_ds.theta_1s, y=calibration_ds.y)
        theta_1s = stack_repeat(theta, len(X))
        pred = cal_model.predict(X=X, theta_1s=theta_1s, log=log)
        if return_calibrated_model:
            return pred, cal_model
        else:
            return pred


def param_scan(
        model: SinglyParameterizedRatioModel,
        X_true: np.ndarray,
        param_grid: ParamGrid,
        verbose: bool = False,
        **predict_params
):
    nllr = []
    iterator = tqdm(param_grid) if verbose else param_grid
    for theta in iterator:
        # predict nllr for individual data points
        theta_1s = stack_repeat(theta, len(X_true))
        logr = model.predict(X_true, theta_1s=theta_1s, log=True, **predict_params)
        # predict nllr over the whole dataset x for each theta
        nllr_pred = -logr.sum()
        nllr.append(nllr_pred)

    nllr = np.array(nllr)
    mle = param_grid[np.argmin(nllr)]
    return _to_meshgrid_shape(nllr, param_grid), mle


def calibrated_param_scan(
        model: SinglyParameterizedRatioModel,
        X_true: np.ndarray,
        param_grid: ParamGrid,
        simulator_func: Callable,
        n_samples_per_theta: int,
        calibration_params: Dict,
        verbose: bool = False,
) -> Sequence[np.ndarray]:
    nllr = []
    iterator = tqdm(param_grid) if verbose else param_grid

    for theta in iterator:
        logr = model.calibrated_predict(
            X=X_true,
            theta=theta,
            n_samples_per_theta=n_samples_per_theta,
            simulator_func=simulator_func,
            calibration_params=calibration_params,
            log=True,
        )
        nllr_pred = -logr.sum()
        nllr.append(nllr_pred)

    nllr = np.array(nllr)
    mle = param_grid[np.argmin(nllr)]
    return _to_meshgrid_shape(nllr, param_grid), mle


def exact_param_scan(
        simulator_func: Callable,
        X_true: np.ndarray,
        param_grid: ParamGrid,
        theta_0: float
) -> Sequence[np.ndarray]:

    p_0 = build_simulator(simulator_func=simulator_func, theta=theta_0)
    log_prob_0 = p_0.log_prob(X_true)

    @tf.function
    def exact_nllr(theta):
        p_theta = build_simulator(simulator_func=simulator_func, theta=theta)
        return -tf.keras.backend.sum(p_theta.log_prob(X_true) - log_prob_0)

    nllr = np.array([exact_nllr(theta.squeeze()).numpy() for theta in param_grid])
    mle = param_grid[np.argmin(nllr)]
    return _to_meshgrid_shape(nllr, param_grid), mle


def _to_meshgrid_shape(arr: np.ndarray, param_grid: ParamGrid):
    meshgrid = param_grid.meshgrid()
    return outer_prod_shape_to_meshgrid_shape(arr, meshgrid[0])
