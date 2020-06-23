from numbers import Number
from typing import Callable, Union, List, Sequence, Tuple

import numpy as np

from active_learning_ratio_estimation.dataset import ParamGrid, SinglyParameterizedRatioDataset, \
    SingleParamIterator, ParamIterator
from active_learning_ratio_estimation.model import SinglyParameterizedRatioModel
from active_learning_ratio_estimation.model.ratio_model import exact_param_scan, param_scan


class ActiveLearner:
    def __init__(self,
                 simulator_func: Callable,
                 X_true: np.ndarray,
                 theta_true: Union[Number, np.ndarray],
                 theta_0: Union[Number, np.ndarray],
                 n_samples_per_theta: int,
                 initial_idx: Sequence[int],
                 ratio_model: SinglyParameterizedRatioModel,
                 total_param_grid: ParamGrid,
                 verbose: bool = True
                 ):
        self.simulator_func = simulator_func
        self.X_true = X_true
        self.theta_true = theta_true
        self.theta_0 = theta_0
        self.n_samples_per_theta = n_samples_per_theta
        self.ratio_model = ratio_model
        self.param_grid = total_param_grid
        self.verbose = verbose

        self._trialed_idx = list(initial_idx)
        self.nllr_predictions = []
        self.mle_predictions = []

        initial_theta_1s = ParamIterator([total_param_grid[idx] for idx in initial_idx])
        self.dataset = SinglyParameterizedRatioDataset.from_simulator(
            simulator_func=simulator_func,
            theta_0=theta_0,
            theta_1_iterator=initial_theta_1s,
            n_samples_per_theta=n_samples_per_theta,
        )

    def fit(self, n_iter: int):
        self.model_fit()
        self.predict()
        for i in range(n_iter):
            iter_msg = f'Active learning iteration {i + 1}/{n_iter}'
            if self.verbose:
                print(iter_msg)
            self._step()

        return self

    def _step(self):
        next_theta_index = self.choose_next_theta_index()
        next_theta = self.all_thetas[next_theta_index]

        new_ds = SinglyParameterizedRatioDataset.from_simulator(
            simulator_func=self.simulator_func,
            theta_0=self.theta_0,
            theta_1_iterator=SingleParamIterator(next_theta),
            n_samples_per_theta=self.n_samples_per_theta,
        )
        self.dataset += new_ds
        self.dataset.shuffle()
        self.model_fit()
        self.predict()
        self._trialed_idx.append(next_theta_index)

    def model_fit(self):
        self.ratio_model.fit(X=self.dataset.x, theta_1s=self.dataset.theta_1s, y=self.dataset.y)

    def choose_next_theta_index(self) -> int:
        next_theta_index = self._choose_next_theta_index()
        if self.verbose:
            print(f'Next theta: {self.all_thetas[next_theta_index]}')
        assert next_theta_index not in self._trialed_idx
        return next_theta_index

    def predict(self):
        mle_estimate = self._predict()
        if self.verbose:
            print(f'MLE estimate: {mle_estimate}')
        return mle_estimate

    @property
    def all_thetas(self):
        return self.param_grid.array

    @property
    def trialed_thetas(self):
        return self.param_grid.array[self._trialed_idx]

    @property
    def untrialed_thetas(self):
        return self.param_grid.array[self._untrialed_idx]

    @property
    def _untrialed_idx(self):
        return np.delete(np.arange(len(self.all_thetas)), self._trialed_idx, axis=0).tolist()

    def _predict(self) -> np.ndarray:
        raise NotImplementedError

    def _choose_next_theta_index(self) -> int:
        raise NotImplementedError


class UpperConfidenceBoundLearner(ActiveLearner):

    def __init__(self,
                 simulator_func: Callable,
                 X_true: np.ndarray,
                 theta_true: Union[Number, np.ndarray],
                 theta_0: Union[Number, np.ndarray],
                 n_samples_per_theta: int,
                 initial_idx: List[int],
                 ratio_model: SinglyParameterizedRatioModel,
                 total_param_grid: ParamGrid,
                 ucb_kappa: float = 1.0,
                 verbose: bool = True
                 ):
        self.nllr_std = []
        self.ucb_kappa = ucb_kappa
        super().__init__(
            simulator_func=simulator_func,
            X_true=X_true,
            theta_true=theta_true,
            theta_0=theta_0,
            n_samples_per_theta=n_samples_per_theta,
            initial_idx=initial_idx,
            ratio_model=ratio_model,
            total_param_grid=total_param_grid,
            verbose=verbose
        )

    def _choose_next_theta_index(self) -> int:
        nllr = self.nllr_predictions[-1]
        std = self.nllr_std[-1]
        acquisition_fn = - nllr + self.ucb_kappa*std
        acquisition_fn[self._trialed_idx] = -np.inf  # don't pick the same point twice
        next_idx = acquisition_fn.argmax()
        return next_idx

    def _predict(self) -> np.ndarray:
        nllr, std, mle = param_scan(
            model=self.ratio_model,
            X_true=self.X_true,
            param_grid=self.param_grid,
            return_std=True,
            to_meshgrid_shape=False,
        )
        self.nllr_predictions.append(nllr)
        self.nllr_std.append(nllr)
        self.mle_predictions.append(mle)
        return mle


class RandomActiveLearner(ActiveLearner):
    # TODO: maybe allow calibration?

    def _choose_next_theta_index(self) -> int:
        return np.random.choice(self._untrialed_idx)

    def _predict(self) -> np.ndarray:
        nllr, mle = param_scan(
            model=self.ratio_model,
            X_true=self.X_true,
            param_grid=self.param_grid,
            return_std=False,
            to_meshgrid_shape=False,
        )
        self.nllr_predictions.append(nllr)
        self.mle_predictions.append(mle)
        return mle