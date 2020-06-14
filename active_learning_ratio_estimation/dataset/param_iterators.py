import itertools
from numbers import Number
from typing import List, Union, Sequence

import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions


class ParamIterator:

    def __init__(self, values: List[np.ndarray]):
        self.values = [val.astype(np.float32) for val in values]

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, item):
        return self.values.__getitem__(item)

    def __len__(self):
        return len(self.values)

    @property
    def array(self):
        return np.array(self.values)


class SingleParamIterator(ParamIterator):

    def __init__(self, theta: Union[Number, np.array], n_samples: int = 1):
        if isinstance(theta, Number):
            theta = np.array([theta])

        values = list(itertools.repeat(theta, n_samples))
        super().__init__(values)


class ParamGrid(ParamIterator):

    def __init__(self, bounds: Sequence[Sequence[float]], num: Union[int, Sequence[int]]):
        if not isinstance(num, Sequence):
            num = [num] * len(bounds)
        self.linspaces = [np.linspace(*bounds_i, num=num[i]).astype(np.float32)
                          for i, bounds_i in enumerate(bounds)]
        values = itertools.product(*self.linspaces)
        values = map(np.array, values)
        values = list(values)
        super().__init__(values)

    def meshgrid(self, **kwargs):
        return np.meshgrid(*self.linspaces, **kwargs)


class DistributionParamIterator(ParamIterator):

    def __init__(self, theta_dist: tfd.Distribution, n_samples: int):
        arr = theta_dist.sample(n_samples).numpy()
        super().__init__([sub_arr for sub_arr in arr])