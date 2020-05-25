import numpy as np


def random(probs):
    return np.random.rand()


def predictive_entropy(probs):
    return - probs*np.log(probs).sum(axis=1)


acquisition_functions = {
    'random': random,
    'entropy': predictive_entropy
}
