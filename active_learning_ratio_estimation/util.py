import itertools
from numbers import Number
from typing import Union

import numpy as np


def _get_likelihoods(x, simulator_func, theta_0, theta_1):
    dist_0 = build_simulator(simulator_func, theta_0)
    dist_1 = build_simulator(simulator_func, theta_1)
    l0 = dist_0.prob(x).numpy()
    l1 = dist_1.prob(x).numpy()
    return l0, l1


def ideal_classifier_probs(x, simulator_func, theta_0, theta_1):
    l0, l1 = _get_likelihoods(x=x, simulator_func=simulator_func, theta_0=theta_0, theta_1=theta_1)
    return l1/(l0+l1)


def likelihood_ratio(x, simulator_func, theta_0, theta_1):
    l0, l1 = _get_likelihoods(x=x, simulator_func=simulator_func, theta_0=theta_0, theta_1=theta_1)
    return l1/l0


def negative_log_likelihood_ratio(x, simulator_func, theta_0, theta_1):
    return -np.log(likelihood_ratio(x=x, simulator_func=simulator_func, theta_0=theta_0, theta_1=theta_1))


def tile_reshape(theta: np.ndarray, reps: int) -> np.ndarray:
    return np.tile(theta, reps).reshape(reps, len(theta))


def outer_prod_shape_to_meshgrid_shape(outer_prod, mesh_grid_arr):
    reshaped = np.zeros_like(mesh_grid_arr)
    for i in range(len(outer_prod)):
        j = i // mesh_grid_arr.shape[0]
        k = i % mesh_grid_arr.shape[0]
        reshaped[k, j] = outer_prod[i]
    return reshaped


def ensure_2d(arr: np.array) -> np.array:
    is2d = len(arr.shape) == 2
    return arr if is2d else arr.reshape(-1, 1)


def ensure_array(item: Union[Number, np.array]):
    is_arr = isinstance(item, np.ndarray)
    return item if is_arr else np.array([item])


def stack_repeat(arr, reps, axis=0):
    return np.stack(list(itertools.repeat(arr, reps)), axis=axis)


def concat_repeat(arr, reps, axis=0):
    return np.concatenate(list(itertools.repeat(arr, reps)), axis=axis)


def build_simulator(simulator_func, theta):
    try:
        simulator = simulator_func(*theta)
    except TypeError:
        simulator = simulator_func(theta)
    return simulator