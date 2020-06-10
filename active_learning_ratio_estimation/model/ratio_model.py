import numpy as np
import tensorflow_probability as tfp
from tqdm import tqdm

from active_learning_ratio_estimation.dataset import ParamGrid, build_singly_parameterized_input, Callable
from active_learning_ratio_estimation.util import estimated_likelihood_ratio, estimated_log_likelihood_ratio, \
    tile_reshape, concat_repeat, outer_prod_shape_to_meshgrid_shape

tfd = tfp.distributions


class BaseRatioModel:
    def __init__(self, clf):
        self.clf = clf

    def _fit(self, X: np.ndarray, y: np.ndarray, **fit_params):
        self.clf.fit(X, y, **fit_params)
        return self

    def _predict(self, X: np.ndarray, log=False, **predict_params):
        y_prob = self.clf.predict_proba(X, **predict_params)[:, 1]
        if log:
            return estimated_log_likelihood_ratio(y_prob)
        else:
            return estimated_likelihood_ratio(y_prob)


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

    def predict_with_calibration(self,
                                 X: np.ndarray,
                                 theta_1: np.ndarray,
                                 n_samples_per_theta: int,
                                 simulator_func: Callable,
                                 calibration_params: Dict,
                                 log=False,
                                 **predict_params
                                 ):


    def predict(self, X: np.ndarray, theta_1s: np.ndarray, log=False, **predict_params):
        model_input = build_singly_parameterized_input(X, theta_1s)
        return self._predict(model_input, log=log, **predict_params)


def param_scan(
        model: SinglyParameterizedRatioModel,
        X_true: np.ndarray,
        param_grid: ParamGrid,
        theta_batch_size: int = 1,
        verbose: bool = False,
        meshgrid_shape: bool = True,
        **predict_params
):
    nllr = []
    theta_groups = np.array_split(param_grid.array, theta_batch_size, axis=0)
    iterator = tqdm(theta_groups) if verbose else theta_groups
    for theta_group in iterator:
        theta_1s = np.concatenate([tile_reshape(theta, reps=len(X_true))
                                  for theta in theta_group], axis=0)
        _X = concat_repeat(X_true, len(theta_group), axis=0)
        # predict nllr for individual data points
        nllr_pred = -model.predict(_X, theta_1s=theta_1s, log=True, **predict_params)
        # predict nllr over the whole dataset x for each theta
        nllr_pred_aggregate = np.stack(np.split(nllr_pred, len(theta_group))).sum(axis=1)
        nllr.append(nllr_pred_aggregate)

    nllr = np.array(nllr)
    if meshgrid_shape:
        meshgrid = param_grid.meshgrid()
        nllr = outer_prod_shape_to_meshgrid_shape(nllr, meshgrid[0])
    return nllr
