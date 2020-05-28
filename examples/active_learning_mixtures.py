import logging

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from active_learning_ratio_estimation.active_learning.active_learner import ActiveLearner
from active_learning_ratio_estimation.dataset import ParamIterator, ParamGrid, SinglyParameterizedRatioDataset
from active_learning_ratio_estimation.model import FlipoutClassifier, SinglyParameterizedRatioModel

# %%


# make logger output only INFO from active_learning_ratio_estimation
logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.INFO
)

for key in logging.Logger.manager.loggerDict:
    if "active_learning_ratio_estimation" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)


# %%


def triple_mixture(gamma):
    mixture_probs = [
        0.5 * (1 - gamma),
        0.5 * (1 - gamma),
        gamma
    ]
    gaussians = [
        tfd.Normal(loc=-2, scale=0.75),
        tfd.Normal(loc=0, scale=2),
        tfd.Normal(loc=1, scale=0.5)
    ]
    dist = tfd.Mixture(
        cat=tfd.Categorical(probs=mixture_probs),
        components=gaussians
    )
    return dist


# %%


theta_bounds = (0, 1)
theta_0 = 0.05
theta_1_iterator = ParamIterator([np.array(bound) for bound in theta_bounds])
num_grid = 101
n_samples_per_theta = int(1e3)
param_grid = ParamGrid(bounds=[theta_bounds], num=num_grid)

# %%
test_param_points = 10
test_iterator = ParamIterator([np.random.rand(1) for _ in range(test_param_points)])
test_dataset = SinglyParameterizedRatioDataset.from_simulator(
    simulator_func=triple_mixture,
    theta_0=theta_0,
    theta_1_iterator=test_iterator,
    n_samples_per_theta=n_samples_per_theta,
    include_log_probs=True
)

# %%
estimator = FlipoutClassifier(
    n_hidden=(15, 15),
    epochs=2,
    patience=0,
    validation_split=0.1,
    verbose=0,
)
ratio_model = SinglyParameterizedRatioModel(estimator=estimator)

# %%

learners = dict()

for acquisition_function in ('entropy', 'random'):
    active_learner = ActiveLearner(simulator_func=triple_mixture,
                                   theta_0=theta_0,
                                   theta_1_iterator=theta_1_iterator,
                                   n_samples_per_theta=n_samples_per_theta,
                                   ratio_model=ratio_model,
                                   total_param_grid=param_grid,
                                   test_dataset=test_dataset)
    active_learner.fit(n_iter=10)
    learners[acquisition_function] = active_learner

# %%


pass
