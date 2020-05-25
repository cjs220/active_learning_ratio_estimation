from typing import Union, Callable

import numpy as np
import pandas as pd

from active_learning_ratio_estimation.active_learning.acquisition_functions import acquisition_functions
from active_learning_ratio_estimation.dataset import ParamGrid, SinglyParameterizedRatioDataset, SingleParamIterator
from active_learning_ratio_estimation.model import SinglyParameterizedRatioModel


class ActiveLearner:

    def __init__(self,
                 ratio_model: SinglyParameterizedRatioModel,
                 initial_dataset: SinglyParameterizedRatioDataset,
                 param_grid: ParamGrid,
                 n_samples_per_theta: int,
                 simulator_func: Callable,
                 test_dataset: SinglyParameterizedRatioDataset = None,
                 acquisition_function: Union[str, Callable] = 'entropy',
                 ):
        self.ratio_model = ratio_model
        self.dataset = initial_dataset
        self.model_fit()
        self.theta_0 = self.dataset.theta_0
        self.param_grid = param_grid
        self.trialed_thetas = [theta for theta in np.unique(initial_dataset.theta_1s)]
        self.n_samples_per_theta = n_samples_per_theta
        self.simulator_func = simulator_func
        self.test_dataset = test_dataset
        self._history = []
        if isinstance(acquisition_function, str):
            acquisition_function = acquisition_functions[acquisition_function]
        self.acquisition_function = acquisition_function

    def model_fit(self):
        self.ratio_model.fit(self.dataset)

    def model_eval(self):
        pass

    @property
    def remaining_thetas(self):
        return list(set(self.param_grid.values) - set(self.trialed_thetas))

    @property
    def history(self):
        return pd.DataFrame(self._history)

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
            self.model_eval()

    def choose_theta(self):
        pass
