import numpy as np
from numpy.testing import assert_array_almost_equal
import tensorflow as tf

from active_learning_ratio_estimation.dataset import SinglyParameterizedRatioDataset, ParamIterator
from active_learning_ratio_estimation.util import stack_repeat


class DummySimulator:

    def __init__(self, theta: np.ndarray):
        self.theta = theta

    def sample(self, n):
        return tf.constant(stack_repeat(self.theta, n))


def test_singly_parameterized_dataset():
    theta_0 = np.array([1.0, 1.0])
    theta_1s = [np.array([0.0, 0.0]), np.array([0.5, 0.5])]
    theta_1_iterator = ParamIterator(theta_1s)
    simulator_func = DummySimulator
    ds = SinglyParameterizedRatioDataset(simulator_func=simulator_func,
                                         theta_0=theta_0,
                                         theta_1_iterator=theta_1_iterator,
                                         n_samples_per_theta=1,
                                         shuffle=False)

    expected = {
        'y': np.array([0, 0, 1, 1]),
        'theta_1s': np.array([
            [0.0, 0.0],
            [0.5, 0.5],
            [0.0, 0.0],
            [0.5, 0.5]
        ]),
        'theta_0s': np.array([
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]),
        'x': np.array([
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.5, 0.5]
        ])
    }

    for arr_name, expected_arr in expected.items():
        arr = getattr(ds, arr_name)
        err_msg = f'Arrays for {arr_name} are not equal;\nexpected=\n{expected_arr}\nactual=\n{arr}'
        assert_array_almost_equal(arr, expected_arr, err_msg=err_msg)
