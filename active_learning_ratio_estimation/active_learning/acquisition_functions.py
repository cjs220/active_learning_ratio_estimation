import numpy as np

from active_learning_ratio_estimation.util import estimated_likelihood_ratio


def predictive_entropy(sampled_probs):
    probs = sampled_probs.mean(axis=1)
    return (- probs*np.log(probs)).sum(axis=1)


def std(sampled_probs):
    return sampled_probs[:, :, 1].std(axis=1)


def std_regressor(sampled_probs):
    s = sampled_probs[:, :, 1]
    r = estimated_likelihood_ratio(s)
    return r.std(axis=1)


acquisition_functions = {
    'entropy': predictive_entropy,
    'std': std,
    'std_regressor': std_regressor
}


if __name__ == '__main__':
    sampled_probs = np.random.rand(1000, 100, 2)
    sampled_probs[:, :, 1] = 1 - sampled_probs[:, :, 0]
    std_regressor(sampled_probs)
