from typing import Union

import numpy as np

from active_learning_ratio_estimation.util import ensure_2d


class Simulator:

    def __init__(self, theta: Union[float, np.ndarray]):
        self.theta = theta

    def sample(self, n):
        raise NotImplementedError

    def log_prob(self, x: np.ndarray):
        raise NotImplementedError


class SimulationDatabase(Simulator):
    """
    Class that contains a set of pre-run simulations in a .npy file, and acts as
    as simulator.
    """

    def __init__(self,
                 theta: Union[float, np.ndarray],
                 filepath: str,
                 with_replacement: bool = False
                 ):
        super().__init__(theta=theta)
        assert filepath.endswith('.npy')
        self.filepath = filepath
        self.with_replacement = with_replacement

        if with_replacement:
            self._sampling_func = self._sample_with_replacement
        else:
            self._counter = 0
            self._sampling_func = self._sample_without_replacement

        data = ensure_2d(np.load(filepath))
        self.data = self._shuffle_data(data)

    def _sample_without_replacement(self, n: int):
        try:
            samples = self.data[self._counter: self._counter + n, :]
        except IndexError:
            raise IndexError(f'Ran out of data in file {self.filepath}')
        self._counter += n
        return samples

    def log_prob(self, x: np.ndarray):
        raise TypeError(f'{self.__class__.__name__} does not support calculation '
                        f'of log probabilities.')

    def sample(self, n: int):
        return self._sampling_func(n)

    def _sample_with_replacement(self, n: int):
        # TODO
        raise RuntimeError

    @staticmethod
    def _shuffle_data(data):
        p = np.random.permutation(len(data))
        return data[p]
