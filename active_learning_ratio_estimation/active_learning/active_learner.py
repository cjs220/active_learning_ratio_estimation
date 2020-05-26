from numbers import Number
from typing import Union, Callable

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF

from active_learning_ratio_estimation.active_learning.acquisition_functions import acquisition_functions
from active_learning_ratio_estimation.dataset import ParamGrid, SinglyParameterizedRatioDataset, SingleParamIterator, \
    ParamIterator
from active_learning_ratio_estimation.model import SinglyParameterizedRatioModel


class ActiveLearner:
    def __init__(self,
                 simulator_func: Callable,
                 theta_0: Union[Number, np.ndarray],
                 theta_1_iterator: ParamIterator,
                 n_samples_per_theta: int,
                 ratio_model: SinglyParameterizedRatioModel,
                 total_param_grid: ParamGrid,
                 test_dataset: SinglyParameterizedRatioDataset = None,
                 acquisition_function: Union[str, Callable] = 'entropy',
                 ucb_kappa: float = 1.0,
                 ):
        self.theta_0 = self.theta_0
        self.n_samples_per_theta = n_samples_per_theta
        self.dataset = SinglyParameterizedRatioDataset(simulator_func=simulator_func,
                                                       theta_0=theta_0,
                                                       theta_1_iterator=theta_1_iterator,
                                                       n_samples_per_theta=n_samples_per_theta)
        self.ratio_model = ratio_model
        self.model_fit()
        self.param_grid = total_param_grid
        self.trialed_thetas = [theta for theta in np.unique(self.dataset.theta_1s)]
        self.simulator_func = simulator_func
        self.test_dataset = test_dataset
        self._test_history = []
        if isinstance(acquisition_function, str):
            acquisition_function = acquisition_functions[acquisition_function]
        self.acquisition_function = acquisition_function
        self.ucb_kappa = ucb_kappa
        self.gp = None
        self.gp_history = []

    def fit(self, n_iter):
        for i in range(n_iter):
            self.step()
        return self

    def model_fit(self):
        self.ratio_model.fit(self.dataset)

    def model_eval(self):
        pred_nllr = self.ratio_model.predict_nllr_dataset(self.test_dataset)
        actual_nllr = self.test_dataset.nllr
        return np.abs(pred_nllr - actual_nllr).mean()

    @property
    def remaining_thetas(self):
        return list(set(self.param_grid.values) - set(self.trialed_thetas))

    @property
    def test_history(self):
        return pd.DataFrame(self._test_history)

    def step(self):
        next_theta = self.choose_theta()
        new_ds = SinglyParameterizedRatioDataset(simulator_func=self.simulator_func,
                                                 theta_0=self.theta_0,
                                                 theta_1_iterator=SingleParamIterator(next_theta),
                                                 n_samples_per_theta=self.n_samples_per_theta)
        self.dataset += new_ds
        self.dataset.shuffle()
        self.model_fit()
        if self.test_dataset is not None:
            hist_item = {
                'theta': next_theta,
                'score': self.model_eval()
            }
            self._test_history.append(hist_item)

    def calculate_marginalised_acquisition(self):
        U_theta = []
        for theta in self.trialed_thetas:
            x = self.dataset.x[self.dataset.theta_1s == theta]
            probs = self.ratio_model.predict_proba(x, theta_1=theta)
            U_theta_x = self.acquisition_function(probs)
            U_theta.append(U_theta_x.mean(axis=1))
        return U_theta

    def choose_theta(self):
        length_scale = np.array([linspace[-1] - linspace[0] for linspace in self.param_grid.linspaces])
        kernel = WhiteKernel() + RBF(length_scale=length_scale)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0, n_restarts_optimizer=5, normalize_y=True)

        X_train = np.array(self.trialed_thetas)
        y_train = np.array(self.calculate_marginalised_acquisition())
        self.gp.fit(X_train, y_train)

        X_test = np.array(self.remaining_thetas)
        mean, std = self.gp.predict(X_test)
        ucb = mean + self.ucb_kappa * std

        self.gp_history.append(dict(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            mean=mean,
            std=std
        ))

        return self.remaining_thetas[np.argmax(ucb)]
