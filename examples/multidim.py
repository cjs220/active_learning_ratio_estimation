#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sys
from functools import partial
from itertools import product

from scipy.stats import chi2

sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import make_sparse_spd_matrix
from corner import corner

from active_learning_ratio_estimation.dataset import RatioDataset
from active_learning_ratio_estimation.util import ideal_classifier_probs, negative_log_likelihood_ratio
from active_learning_ratio_estimation.model import RegularRatioModel, BayesianRatioModel

# get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(0)
tf.random.set_seed(0)


#%%

class MultiDimToyModel(tfd.TransformedDistribution):

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

        # compose linear transform
        R = make_sparse_spd_matrix(5, alpha=0.5, random_state=7).astype(np.float32)
        self.R = R
        transform = tf.linalg.LinearOperatorFullMatrix(R)
        bijector = tfp.bijectors.AffineLinearOperator(scale=transform)

        super().__init__(distribution=self.z_distribution, bijector=bijector)

    @property
    def z_distribution(self):
        z_distribution = tfd.Blockwise([
            tfd.Normal(loc=self.alpha, scale=1),  # z1
            tfd.Normal(loc=self.beta, scale=3),  # z2
            tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=[0.5, 0.5]),
                components_distribution=tfd.Normal(
                    loc=[-2, 2],
                    scale=[1, 0.5]
                )
            ),  # z3
            tfd.Exponential(3),  # z4
            tfd.Exponential(0.5),  # z5
        ])
        return z_distribution


# In[7]:


# Plot histograms / correlations of true distributions
true_alpha = 1
true_beta = -1
p_true = MultiDimToyModel(alpha=1, beta=-1)
X_true = p_true.sample(500)
# fig = corner(X_true, bins=20, smooth=0.85, labels=["X0", "X1", "X2", "X3", "X4"])
# plt.show()


# In[8]:


# find exact maximum likelihood
var_alpha = tf.Variable(tf.constant(0, dtype=tf.float32))
var_beta = tf.Variable(tf.constant(0, dtype=tf.float32))
p_var = MultiDimToyModel(var_alpha, var_beta)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
n_iter = int(1e3)
nll = tf.function(lambda: -tf.keras.backend.sum(p_var.log_prob(X_true)))

for i in range(n_iter):
    optimizer.minimize(nll, [var_alpha, var_beta])


alpha_mle = var_alpha.numpy()
beta_mle = var_beta.numpy()
max_log_prob = p_var.log_prob(X_true)

print(f'Exact MLE: alpha={alpha_mle}, beta={beta_mle}')


# %%


# Calculate contours of exact negative log likelihood ratio

# @tf.function
# def nllr_exact(alpha, beta, X):
#     p_theta = MultiDimToyModel(alpha=alpha, beta=beta)
#     return -tf.keras.backend.sum((p_theta.log_prob(X) - max_log_prob))


n_grid = 10
alpha_bounds = (0.75, 1.25)
beta_bounds = (-2, 0)
alphas = np.linspace(alpha_bounds[0], alpha_bounds[1], n_grid).astype(np.float32)
betas = np.linspace(beta_bounds[0], beta_bounds[1], n_grid).astype(np.float32)
# Alphas, Betas = np.meshgrid(alphas, betas)
#
# exact_contours = np.zeros_like(Alphas)
# for i in range(n_grid):
#     for j in range(n_grid):
#         alpha = tf.constant(Alphas[i, j])
#         beta = tf.constant(Betas[i, j])
#         nllr = nllr_exact(alpha, beta, X_true)
#         exact_contours[i, j] = nllr


#%%


# plot contours
# plt.figure()
# plt.contour(Alphas, Betas, exact_contours,
#             levels=[chi2.ppf(0.683, df=2),
#                     chi2.ppf(0.9545, df=2),
#                     chi2.ppf(0.9973, df=2)], colors=["w"])
# plt.contourf(Alphas, Betas, exact_contours, 50, vmin=0, vmax=30)
# plt.plot([true_alpha], [true_beta],'ro', markersize=8, label='True')
# plt.plot([alpha_mle], [beta_mle], 'go', markersize=8, label='MLE')
# plt.xlabel(r"$\alpha$")
# plt.ylabel(r"$\beta$")
# plt.title(r'$-2\log \left[ L(\theta) / L(\theta _\mathrm{MLE}) \right]$')
# plt.xlim(alpha_bounds)
# plt.ylim(beta_bounds)
# cb = plt.colorbar()
# plt.legend(loc='upper center', fancybox=True, shadow=True)
# plt.show()


#%%


# create dataset for fitting
theta_1 = np.array(list(product(alphas, betas)))
theta_0 = np.array([(alpha_mle, beta_mle) for _ in range(len(theta_1))])

ds = RatioDataset(n_samples_per_theta=10,
                  simulator_func=MultiDimToyModel,
                  theta_0_dist=theta_0,
                  theta_1_dist=theta_1)

print(ds)
pass
