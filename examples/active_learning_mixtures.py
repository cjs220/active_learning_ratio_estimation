import numpy as np

from active_learning_ratio_estimation.active_learning.active_learner import ActiveLearner
from active_learning_ratio_estimation.dataset import ParamIterator, ParamGrid, SinglyParameterizedRatioDataset
from active_learning_ratio_estimation.model import FlipoutClassifier, SinglyParameterizedRatioModel
from .mixtures import triple_mixture


#%%


theta_bounds = (0, 1)
theta_0 = 0.05
theta_1_iterator = ParamIterator([np.array(bound) for bound in theta_bounds])
num_grid = 101
n_samples_per_theta = int(1e3)
param_grid = ParamGrid(bounds=[theta_bounds], num=num_grid)


# %%
test_param_points = 10
test_iterator = ParamIterator([np.random.rand() for _ in range(test_param_points)])
test_dataset = SinglyParameterizedRatioDataset(simulator_func=triple_mixture,
                                               theta_0=theta_0,
                                               theta_1_iterator=test_iterator,
                                               n_samples_per_theta=n_samples_per_theta)


#%%
estimator = FlipoutClassifier(n_hidden=(15, 15),
                              epochs=2,
                              patience=0,
                              validation_split=0.1)
ratio_model = SinglyParameterizedRatioModel(estimator=estimator)


#%%

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


#%%


pass