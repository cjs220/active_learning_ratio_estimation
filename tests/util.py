import numpy as np
import tensorflow as tf
from numpy.testing import assert_array_almost_equal

from active_learning_ratio_estimation.util import stack_repeat


class DummySimulator:

    def __init__(self, theta: np.ndarray):
        self.theta = theta

    def sample(self, n):
        # returns theta n times
        return tf.constant(stack_repeat(self.theta, n))

    def log_prob(self, x: np.array):
        # returns 1 if x==theta, 0.5 else
        return tf.constant(0.5*(x == self.theta)[:, 0] + 0.5)


def assert_named_array_equal(expected_arr, arr, arr_name):
    err_msg = f'Arrays for {arr_name} are not equal;\nexpected=\n{expected_arr}\nactual=\n{arr}'
    assert_array_almost_equal(arr, expected_arr, err_msg=err_msg)
