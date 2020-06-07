from typing import Callable, List

import numpy as np
import tensorflow_probability as tfp
from joblib import Parallel, delayed

from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from active_learning_ratio_estimation.dataset import RatioDataset, SinglyParameterizedRatioDataset, \
    UnparameterizedRatioDataset, build_unparameterized_input, build_singly_parameterized_input, ParamGrid, \
    SingleParamIterator
from active_learning_ratio_estimation.util import tile_reshape, outer_prod_shape_to_meshgrid_shape, concat_repeat, \
    stack_repeat, estimated_likelihood_ratio

tfd = tfp.distributions


class RatioModel:
    def __init__(self,
                 estimator,
                 calibration_method=None,
                 cv=1,
                 normalize_input=True
                 ):
        self.calibration_method = calibration_method
        if calibration_method is not None:
            estimator = CalibratedClassifierCV(base_estimator=estimator,
                                               method=self.calibration_method,
                                               cv=cv)

        steps = [('clf', estimator)]
        if normalize_input:
            steps.insert(0, ('StandardScaler', StandardScaler()))
        self.estimator = Pipeline(steps=steps)

    @property
    def keras_model_(self):
        return self.estimator.steps[-1][1].model_

    def fit(self, dataset: RatioDataset):
        model_input = dataset.build_input()
        self.estimator.fit(model_input, dataset.y)
        return self

    def predict(self, *args):
        raise NotImplementedError

    def predict_proba(self, *args):
        raise NotImplementedError

    def predict_likelihood_ratio(self, x, *args):
        probs = self.predict_proba(x, *args)[:, 1]
        return estimated_likelihood_ratio(probs)

    def predict_negative_log_likelihood_ratio(self, x, *args):
        return -np.log(self.predict_likelihood_ratio(x, *args))

    def _call_on_dataset(self, method: str, dataset: RatioDataset):
        model_input = dataset.build_input()
        return getattr(self.estimator, method)(model_input)

    def predict_dataset(self, dataset: RatioDataset):
        return self._call_on_dataset('predict', dataset)

    def predict_proba_dataset(self, dataset: RatioDataset):
        return self._call_on_dataset('predict_proba', dataset)

    def predict_likelihood_ratio_dataset(self, dataset: RatioDataset):
        probs = self.predict_proba_dataset(dataset)
        return estimated_likelihood_ratio(probs)

    def predict_nllr_dataset(self, dataset):
        return -np.log(self.predict_likelihood_ratio_dataset(dataset))


class UnparameterizedRatioModel(RatioModel):

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


class SinglyParameterizedRatioModel(RatioModel):

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
        if not len(theta_1) == len(x):
            assert theta_1.shape == self.theta_0_.shape
            theta_1 = stack_repeat(theta_1, len(x))
        model_input = build_singly_parameterized_input(x=x, theta_1s=theta_1)
        return self.estimator.predict_proba(model_input)

    def predict_dataset(self, dataset: SinglyParameterizedRatioDataset):
        assert np.all(self.theta_0_ == dataset.theta_0)
        return super().predict_dataset(dataset)

    def predict_proba_dataset(self, dataset: SinglyParameterizedRatioDataset):
        assert np.all(self.theta_0_ == dataset.theta_0)
        return super().predict_proba_dataset(dataset)

    def _param_scan(self,
                    x: np.ndarray,
                    param_grid: ParamGrid,
                    meshgrid_shape: bool = True,
                    get_model: Callable = None,
                    verbose: bool = False
                    ):

        def _calc_nllr(theta):
            theta_1 = tile_reshape(theta, reps=len(x))
            model = get_model(theta) if get_model is not None else self
            # predict nllr for individual data points
            nllr_pred = model.predict_negative_log_likelihood_ratio(x, theta_1)
            # predict nllr over the whole dataset
            return nllr_pred.sum()

        nllr = []
        iterator = tqdm(param_grid) if verbose else param_grid
        for theta in iterator:
            nllr.append(_calc_nllr(theta))

        nllr = np.array(nllr)
        mle = param_grid[np.argmin(nllr)]
        if meshgrid_shape:
            meshgrid = param_grid.meshgrid()
            nllr = outer_prod_shape_to_meshgrid_shape(nllr, meshgrid[0])
        return nllr, mle

    def nllr_param_scan(self,
                        x: np.ndarray,
                        param_grid: ParamGrid,
                        meshgrid_shape: bool = True,
                        verbose: bool = False
                        ):
        nllr, mle = self._param_scan(
            get_model=None,
            x=x,
            param_grid=param_grid,
            meshgrid_shape=meshgrid_shape,
            verbose=verbose
        )
        return nllr, mle

    def nllr_param_scan_with_calibration(self,
                                         x: np.ndarray,
                                         param_grid: ParamGrid,
                                         n_samples_per_theta: int,
                                         simulator_func: Callable,
                                         calibration_method: str = 'sigmoid',
                                         meshgrid_shape: bool = True,
                                         verbose: bool = False
                                         ):
        # like nllr param scan, but with calibration at each parameter point
        assert self.calibration_method is None

        def _calibrate_on_predict(theta):
            new_model = self.__class__(
                estimator=self.estimator,
                calibration_method=calibration_method,
                normalize_input=False,
                cv='prefit'
            )
            new_ds = SinglyParameterizedRatioDataset.from_simulator(
                simulator_func=simulator_func,
                theta_0=self.theta_0_,
                theta_1_iterator=SingleParamIterator(theta),
                n_samples_per_theta=n_samples_per_theta
            )
            new_model.fit(new_ds)
            return new_model

        nllr, mle = self._param_scan(
            get_model=_calibrate_on_predict,
            x=x,
            param_grid=param_grid,
            meshgrid_shape=meshgrid_shape,
            verbose=verbose
        )

        return nllr, mle
