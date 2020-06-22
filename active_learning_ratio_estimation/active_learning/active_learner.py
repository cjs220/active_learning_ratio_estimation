from numbers import Number
from typing import Callable, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from active_learning_ratio_estimation.dataset import ParamIterator, ParamGrid, SinglyParameterizedRatioDataset
from active_learning_ratio_estimation.model import SinglyParameterizedRatioModel


class ActiveLearner:
    def __init__(self,
                 simulator_func: Callable,
                 theta_0: Union[Number, np.ndarray],
                 theta_1_iterator: ParamIterator,
                 n_samples_per_theta: int,
                 ratio_model: SinglyParameterizedRatioModel,
                 total_param_grid: ParamGrid,
                 ucb_kappa: float = 1.0,
                 ):
        self._train_history = []
        self._test_history = []
        self.theta_0 = theta_0
        self.n_samples_per_theta = n_samples_per_theta
        # logger.info('Initialised ActiveLeaner; simulating initial dataset.')
        self.dataset = SinglyParameterizedRatioDataset.from_simulator(
            simulator_func=simulator_func,
            theta_0=theta_0,
            theta_1_iterator=theta_1_iterator,
            n_samples_per_theta=n_samples_per_theta,
        )
        self.ratio_model = ratio_model
        self.model_fit()
        self.param_grid = total_param_grid
        trialed_mask = np.array([np.array(total_param_grid.values) == theta
                                 for theta in theta_1_iterator]).all(axis=2).sum(axis=0).astype(bool)
        self._trialed_idx = np.arange(len(total_param_grid))[trialed_mask].tolist()
        self.simulator_func = simulator_func
        self.ucb_kappa = ucb_kappa

    def model_fit(self):
        self.ratio_model.fit(X=np.ndarray)