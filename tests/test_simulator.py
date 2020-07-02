import os
from tempfile import gettempdir

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from active_learning_ratio_estimation.simulator import SimulationDatabase

tfd = tfp.distributions


class GaussianSimulationDatabase(SimulationDatabase):

    def __init__(self, theta: float):
        super().__init__(theta=theta,
                         filepath=_file_path(theta),
                         with_replacement=False)


def _file_path(theta: float):
    return os.path.join(gettempdir(), f'{theta}.npy')


def test_simulation_database():
    # Test loading samples from npy file
    thetas = np.linspace(0, 1, 6)

    for theta in thetas:
        dist = tfd.Normal(loc=theta, scale=1)
        x = dist.sample(10)
        np.save(_file_path(theta), x)

    sim = GaussianSimulationDatabase(thetas[0])
    samples = sim.sample(5)