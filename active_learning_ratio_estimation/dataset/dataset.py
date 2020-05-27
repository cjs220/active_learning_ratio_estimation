from numbers import Number
from typing import Callable, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from active_learning_ratio_estimation.dataset.param_iterators import ParamIterator
from active_learning_ratio_estimation.util import ensure_2d, ensure_array, stack_repeat, build_simulator, concat_repeat

tfd = tfp.distributions


def build_unparameterized_input(x):
    return ensure_2d(x)


def build_singly_parameterized_input(x, theta_1s):
    return np.concatenate([ensure_2d(x), ensure_2d(theta_1s)], axis=1)


class RatioDataset:
    _possibly_none_attrs = ('y', 'log_prob_0', 'log_prob_1')

    def __init__(self,
                 x: np.array,
                 theta_0s: np.array,
                 theta_1s: np.array,
                 y: np.array = None,
                 log_prob_0: np.array = None,
                 log_prob_1: np.array = None,
                 shuffle: bool = True
                 ):
        self.x = ensure_2d(x)
        self.theta_0s = theta_0s
        self.theta_1s = theta_1s
        self.y = y
        self.log_prob_0 = log_prob_0
        self.log_prob_1 = log_prob_1

        # check arrays have same length
        arrs = [self.x, self.theta_0s, self.theta_1s]
        for arr in (y, log_prob_0, log_prob_1):
            if arr is not None:
                arrs.append(arr)
        if len(set(map(len, arrs))) != 1:
            raise ValueError('Arrays have different lengths')

        if shuffle:
            self.shuffle()

    def shuffle(self):
        p = np.random.permutation(len(self.x))
        self.x = self.x[p]
        self.theta_0s = self.theta_0s[p]
        self.theta_1s = self.theta_1s[p]

        # shuffle y and nllr if they have been given
        for arr_name in self._possibly_none_attrs:
            try:
                arr = getattr(self, arr_name)
                setattr(self, arr_name, arr[p])
            except TypeError:
                # arr is None
                pass

    def build_input(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.x)

    def __add__(self, other):

        def _concat_attr(attr_name):
            this_one = getattr(self, attr_name)
            that_one = getattr(other, attr_name)
            return np.concatenate([this_one, that_one], axis=0)

        kwargs = {}
        for attr_name in ('x', 'theta_0s', 'theta_1s'):
            kwargs[attr_name] = _concat_attr(attr_name)

        for attr_name in self._possibly_none_attrs:
            try:
                kwargs[attr_name] = _concat_attr(attr_name)
            except ValueError:
                # attr is None
                pass
        return self.__class__(**kwargs)


class UnparameterizedRatioDataset(RatioDataset):

    def __init__(self,
                 theta_0,
                 theta_1,
                 x: np.array,
                 y: np.array = None,
                 log_prob_0: np.array = None,
                 log_prob_1: np.array = None,
                 shuffle: bool = True
                 ):
        self.theta_0 = theta_0
        self.theta_1 = theta_1
        theta_0s = stack_repeat(theta_0, len(x))
        theta_1s = stack_repeat(theta_1, len(x))
        super().__init__(x=x,
                         y=y,
                         theta_0s=theta_0s,
                         theta_1s=theta_1s,
                         log_prob_0=log_prob_0,
                         log_prob_1=log_prob_1,
                         shuffle=shuffle)

    @classmethod
    def from_simulator(cls,
                       simulator_func: Callable,
                       theta_0: Union[Number, np.array],
                       theta_1: Union[Number, np.array],
                       n_samples_per_theta: int,
                       shuffle: bool = True,
                       include_log_probs: bool = False,
                       ):
        theta_0, theta_1 = map(ensure_array, [theta_0, theta_1])
        assert len(theta_0.shape) == 1 == len(theta_1.shape)

        sim0 = build_simulator(simulator_func, theta_0)
        sim1 = build_simulator(simulator_func, theta_1)
        x0 = sim0.sample(n_samples_per_theta).numpy()
        x1 = sim1.sample(n_samples_per_theta).numpy()
        y0 = np.zeros(len(x0))
        y1 = np.ones_like(y0)
        x = np.concatenate([x0, x1], axis=0)
        y = np.concatenate([y0, y1], axis=0)

        if include_log_probs:
            log_prob_0 = sim0.log_prob(x)
            log_prob_1 = sim1.log_prob(x)
        else:
            log_prob_0 = None
            log_prob_1 = None

        return cls(x=x,
                   y=y,
                   theta_0=theta_0,
                   theta_1=theta_1,
                   log_prob_0=log_prob_0,
                   log_prob_1=log_prob_1,
                   shuffle=shuffle)

    def build_input(self):
        return build_unparameterized_input(self.x)


class SinglyParameterizedRatioDataset(RatioDataset):

    def __init__(self,
                 theta_0,
                 theta_1s,
                 x: np.array,
                 y: np.array = None,
                 log_prob_0: np.array = None,
                 log_prob_1: np.array = None,
                 shuffle: bool = True
                 ):
        self.theta_0 = theta_0
        theta_0s = stack_repeat(theta_0, len(x))
        super().__init__(x=x,
                         y=y,
                         theta_0s=theta_0s,
                         theta_1s=theta_1s,
                         log_prob_0=log_prob_0,
                         log_prob_1=log_prob_1,
                         shuffle=shuffle)

    @classmethod
    def from_simulator(cls,
                       simulator_func: Callable,
                       theta_0: Union[Number, np.ndarray],
                       theta_1_iterator: ParamIterator,
                       n_samples_per_theta: int,
                       shuffle: bool = True,
                       include_log_probs: bool = False
                       ):
        theta_0 = ensure_array(theta_0)
        assert len(theta_0.shape) == 1

        sim0 = build_simulator(simulator_func, theta_0)
        x0 = sim0.sample(n_samples_per_theta * len(theta_1_iterator)).numpy()
        x0 = ensure_2d(x0)
        y0 = np.zeros(len(x0))
        y1 = np.ones_like(y0)
        x1 = []
        theta_1s = np.zeros((len(x0), len(theta_0)))

        if include_log_probs:
            log_prob_0_x0 = []
            log_prob_0_x1 = []
            log_prob_1_x0 = []
            log_prob_1_x1 = []

        # @tf.function
        def _simulate(theta_1, x0_i):
            sim1 = build_simulator(simulator_func, theta_1)
            x1_i = sim1.sample(n_samples_per_theta).numpy()
            x1_i = ensure_2d(x1_i)
            x0_i = ensure_2d(x0_i)
            x1.append(x1_i)
            if include_log_probs:
                log_prob_0_x0.append(sim0.log_prob(x0_i))
                log_prob_0_x1.append(sim0.log_prob(x1_i))
                log_prob_1_x0.append(sim1.log_prob(x0_i))
                log_prob_1_x1.append(sim1.log_prob(x1_i))

        for i, theta_1 in enumerate(theta_1_iterator):
            start = i * n_samples_per_theta
            stop = (i + 1) * n_samples_per_theta
            theta_1s[start:stop, :] = theta_1
            _simulate(theta_1, x0[start:stop, :])

        if include_log_probs:
            log_prob_0 = np.concatenate(log_prob_0_x0 + log_prob_0_x1, axis=0)
            log_prob_1 = np.concatenate(log_prob_1_x0 + log_prob_1_x1, axis=0)
        else:
            log_prob_0 = None
            log_prob_1 = None

        x1 = ensure_2d(np.concatenate(x1, axis=0))
        x = np.concatenate([x0, x1], axis=0)
        y = np.concatenate([y0, y1], axis=0)
        theta_1s = concat_repeat(theta_1s, 2, axis=0)
        return cls(x=x,
                   y=y,
                   theta_0=theta_0,
                   theta_1s=theta_1s,
                   log_prob_0=log_prob_0,
                   log_prob_1=log_prob_1,
                   shuffle=shuffle)

    def build_input(self):
        return build_singly_parameterized_input(x=self.x, theta_1s=self.theta_1s)
