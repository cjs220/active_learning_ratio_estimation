import numpy as np
import pytest

from active_learning_ratio_estimation.dataset import SinglyParameterizedRatioDataset, ParamIterator, \
    UnparameterizedRatioDataset
from tests.util import DummySimulator, assert_named_array_equal


def test_unparameterized_dataset():
    theta_0 = np.array([1.0, 1.0])
    theta_1 = np.array([0.0, 0.0])
    n_samples_per_theta = 1
    ds = UnparameterizedRatioDataset.from_simulator(
        simulator_func=DummySimulator,
        theta_0=theta_0,
        theta_1=theta_1,
        n_samples_per_theta=n_samples_per_theta,
        shuffle=False,
        include_log_probs=True
    )
    expected = {
        'y': np.array([0, 1]),
        'theta_0s': np.array([
            [1.0, 1.0],
            [1.0, 1.0]
        ]),
        'theta_1s': np.array([
            [0.0, 0.0],
            [0.0, 0.0]
        ]),
        'x': np.array([
            [1.0, 1.0],
            [0.0, 0.0]
        ]),
        'log_prob_0': np.array([1, 0.5]),
        'log_prob_1': np.array([0.5, 1]),

    }

    for arr_name, expected_arr in expected.items():
        arr = getattr(ds, arr_name)
        assert_named_array_equal(expected_arr=expected_arr, arr=arr, arr_name=arr_name)


class TestSinglyParameterizedDataset:

    def test_from_simulator(self):
        theta_0 = np.array([1.0, 1.0])
        theta_1s = [np.array([0.0, 0.0]), np.array([0.5, 0.5])]
        theta_1_iterator = ParamIterator(theta_1s)
        ds = SinglyParameterizedRatioDataset.from_simulator(
            simulator_func=DummySimulator,
            theta_0=theta_0,
            theta_1_iterator=theta_1_iterator,
            n_samples_per_theta=1,
            shuffle=False,
            include_log_probs=True
        )

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
            ]),
            'log_prob_0': np.array([1, 1, 0.5, 0.5]),
            'log_prob_1': np.array([0.5, 0.5, 1, 1])
        }
        for arr_name, expected_arr in expected.items():
            arr = getattr(ds, arr_name)
            assert_named_array_equal(expected_arr=expected_arr, arr=arr, arr_name=arr_name)

    def test_add(self):
        zeros = np.array([0, 0])
        ones = np.array([1, 1])
        theta_0 = 0.5
        ds_a = SinglyParameterizedRatioDataset(theta_0=theta_0, theta_1s=zeros, x=zeros, y=zeros)
        ds_b = SinglyParameterizedRatioDataset(theta_0=theta_0, theta_1s=ones, x=ones, y=ones)
        ds = ds_a + ds_b
        for arr_name in ('x', 'y', 'theta_1s'):
            expected = np.array([0, 0, 1, 1])
            assert_named_array_equal(expected, getattr(ds, arr_name).squeeze(), arr_name)

        assert ds.log_prob_0 is None and ds.log_prob_1 is None
