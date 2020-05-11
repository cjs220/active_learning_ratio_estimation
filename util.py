import numpy as np


def _get_likelihoods(x, simulator_func, theta_0, theta_1):
    dist_0 = simulator_func(theta_0)
    dist_1 = simulator_func(theta_1)
    l0 = dist_0.prob(x)
    l1 = dist_1.prob(x)
    return l0, l1


def ideal_classifier_probs(x, simulator_func, theta_0, theta_1):
    l0, l1 = _get_likelihoods(x=x, simulator_func=simulator_func, theta_0=theta_0, theta_1=theta_1)
    return l1/(l0+l1)


def likelihood_ratio(x, simulator_func, theta_0, theta_1):
    l0, l1 = _get_likelihoods(x=x, simulator_func=simulator_func, theta_0=theta_0, theta_1=theta_1)
    return l1/l0


def negative_log_likelihood_ratio(x, simulator_func, theta_0, theta_1):
    return -np.log(likelihood_ratio(x=x, simulator_func=simulator_func, theta_0=theta_0, theta_1=theta_1))
