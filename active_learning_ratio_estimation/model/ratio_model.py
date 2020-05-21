import numpy as np
import tensorflow_probability as tfp

from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow_core.python.keras.wrappers.scikit_learn import KerasClassifier

from active_learning_ratio_estimation.dataset import RatioDataset, SinglyParameterizedRatioDataset, \
    UnparameterizedRatioDataset, build_unparameterized_input, build_singly_parameterized_input, ParamGrid
from active_learning_ratio_estimation.util import tile_reshape, outer_prod_shape_to_meshgrid_shape, concat_repeat, \
    stack_repeat

tfd = tfp.distributions


class BaseRatioModel:
    def __init__(self,
                 build_fn,
                 fit_kwargs=None,
                 build_fn_kwargs=None,
                 calibration_method=None,
                 cv=1,
                 normalize_input=True):

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

    def predict_negative_log_likelihood_ratio(self, x, *args):
        # y_pred = self.predict_proba(x, *args)[:, 1]
        # return - (np.log(y_pred) - np.log(1 - y_pred))
        return -np.log(self.predict_likelihood_ratio(x, *args))


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
        if not len(theta_1) == len(x):
            assert theta_1.shape == self.theta_0_.shape
            theta_1 = stack_repeat(theta_1, len(x))
        model_input = build_singly_parameterized_input(x=x, theta_1s=theta_1)
        return self.estimator.predict_proba(model_input)

    def nllr_param_scan(self, x: np.ndarray, param_grid: ParamGrid, meshgrid_shape: bool = True):
        # calculate the negative likelihood ratio across parameter grid for given x
        theta_1 = np.concatenate([tile_reshape(theta, reps=len(x)) for theta in param_grid.values], axis=0)
        x_ = concat_repeat(x, len(param_grid), axis=0)
        nllr_pred = self.predict_negative_log_likelihood_ratio(x_, theta_1)
        nllr_pred_total = np.stack(np.split(nllr_pred, len(param_grid.values))).sum(axis=1)
        if meshgrid_shape:
            meshgrid = param_grid.meshgrid()
            nllr_pred_total = outer_prod_shape_to_meshgrid_shape(nllr_pred_total, meshgrid[0])
        return nllr_pred_total

