from numbers import Number
from typing import List, Dict, Callable, Union
from random import shuffle

import numpy as np
import pandas as pd
import tensorflow_probability as tfp

tfd = tfp.distributions


class DataSet:

    def __init__(self, samples: List[Dict[str, np.array]]):
        self.samples = samples

    def __iter__(self):
        return self.samples.__iter__()

    def __add__(self, other):
        return self.__class__(self.samples + other.samples)

    def __len__(self):
        return len(self.samples)

    @property
    def x(self):
        return np.array([sample['x'] for sample in self.samples])

    @property
    def y(self):
        return np.array([sample['y'] for sample in self.samples])

    @property
    def theta_0(self):
        return np.array([sample['theta_0'] for sample in self.samples])

    @property
    def theta_1(self):
        return np.array([sample['theta_1'] for sample in self.samples])

    def shuffle(self):
        shuffle(self.samples)

    def filter(self, bool_arr):
        filtered_df = self.dataframe[bool_arr]
        return self.from_dataframe(filtered_df)

    @property
    def dataframe(self):
        return pd.DataFrame(self.samples)

    def to_csv(self, path_or_buf):
        self.dataframe.to_csv(path_or_buf, index=False)

    @classmethod
    def from_csv(cls, path_or_buf):
        # TODO: fix for multidim x
        df = pd.read_csv(path_or_buf)
        return cls.from_dataframe(df)

    @classmethod
    def from_dataframe(cls, df):
        samples = [x[1].to_dict() for x in df.iterrows()]
        return DataSet(samples)


class RatioDataset(DataSet):

    def __init__(self,
                 n_samples_per_theta: int,
                 simulator_func: Callable,
                 theta_0_dist: Union[tfd.Distribution, Number, np.array],
                 theta_1_dist: Union[tfd.Distribution, Number, np.array],
                 n_thetas: int = None):

        theta_0s = self._choose_theta(theta_0_dist, n_thetas)
        theta_1s = self._choose_theta(theta_1_dist, n_thetas)

        samples = []
        for theta_0, theta_1 in zip(theta_0s, theta_1s):
            for theta, y in [(theta_0, 0), (theta_1, 1)]:
                x = self.simulate(simulator_func=simulator_func, theta=theta, n_samples_per_theta=n_samples_per_theta)
                new_samples = [dict(x=x_i, theta_0=theta_0, theta_1=theta_1, y=y) for x_i in x]
                samples += new_samples
        shuffle(samples)

        super().__init__(samples)

    @staticmethod
    def _choose_theta(dist, n_samples) -> np.array:
        if hasattr(dist, 'sample'):
            theta = dist.sample(n_samples).numpy()
        elif isinstance(dist, Number):
            # TODO: remove this
            theta = np.array([dist] * n_samples)
        else:
            # theta is an array of shape (num_thetas, -1)
            assert len(dist.shape) == 2
            assert n_samples is None
            theta = dist
        return theta

    @staticmethod
    def simulate(simulator_func, theta, n_samples_per_theta):
        simulator = simulator_func(*theta)
        x = simulator.sample(n_samples_per_theta).numpy()
        return x
