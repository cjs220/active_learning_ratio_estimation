#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sys

from scipy.stats import chi2

from active_learning_ratio_estimation.model import SinglyParameterizedRatioModel, build_feedforward,\
    build_bayesian_flipout
from active_learning_ratio_estimation.util import negative_log_likelihood_ratio, ideal_classifier_probs

sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import make_sparse_spd_matrix

from active_learning_ratio_estimation.dataset import ParamGrid, SinglyParameterizedRatioDataset

# get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(0)
tf.random.set_seed(0)


# %%

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
theta_mle = np.array([alpha_mle, beta_mle])
max_log_prob = p_var.log_prob(X_true)

print(f'Exact MLE: alpha={alpha_mle}, beta={beta_mle}')


# %%


# Calculate contours of exact negative log likelihood ratio

@tf.function
def nllr_exact(alpha, beta, X):
    p_theta = MultiDimToyModel(alpha=alpha, beta=beta)
    return -tf.keras.backend.sum((p_theta.log_prob(X) - max_log_prob))


num = 10
alpha_bounds = (0.75, 1.25)
beta_bounds = (-2, 0)
param_grid = ParamGrid(bounds=[alpha_bounds, beta_bounds], num=num)
Alphas, Betas = param_grid.meshgrid()

exact_contours = np.zeros_like(Alphas)
for i in range(num):
    for j in range(num):
        alpha = tf.constant(Alphas[i, j])
        beta = tf.constant(Betas[i, j])
        nllr = nllr_exact(alpha, beta, X_true)
        exact_contours[i, j] = nllr

# %%


# create dataset for fitting
theta_0 = np.array([alpha_mle, beta_mle])
ds = SinglyParameterizedRatioDataset(
    simulator_func=MultiDimToyModel,
    theta_0=theta_0,
    theta_1_iterator=param_grid,
    n_samples_per_theta=int(1e3)
)

# %%


# hyperparams
epochs = 2
patience = 2
validation_split = 0.1
n_hidden = (40, 40)
n_samples = int((1 - validation_split) * len(ds))

fit_kwargs = dict(
    epochs=epochs,
    validation_split=validation_split,
    verbose=2,
    # callbacks=[tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=patience)],
)

# regular, calibrated model
# cv = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=1)
# regular_calibrated = SinglyParameterizedRatioModel(
#     build_fn=build_feedforward,
#     build_fn_kwargs=dict(n_hidden=n_hidden, activation='tanh'),
#     fit_kwargs=fit_kwargs,
#     calibration_method='sigmoid',
#     cv=cv,
# )

# bayesian, uncalibrated model
bayesian_uncalibrated = SinglyParameterizedRatioModel(
    build_fn=build_bayesian_flipout,
    build_fn_kwargs=dict(n_hidden=n_hidden, activation='relu', n_samples=n_samples),
    fit_kwargs=fit_kwargs,
    calibration_method=None,
)


def fit_predict(clf):
    clf.fit(ds)
    theta_1 = np.array([0.5, 0.5]).astype(np.float32)
    p_pred = clf.predict_proba(X_true, theta_1=theta_1)[:, 1]
    p_ideal = ideal_classifier_probs(simulator_func=MultiDimToyModel,
                                     x=X_true,
                                     theta_0=theta_0,
                                     theta_1=theta_1)
    average_diff = np.abs(p_pred - p_ideal).mean()
    nllr_pred = clf.nllr_param_scan(x=X_true, param_grid=param_grid)
    return nllr_pred


models = {
    # 'Regular Uncalibrated': regular_uncalibrated,
    'Bayesian Uncalibrated': bayesian_uncalibrated,
    # 'Regular Calibrated': regular_calibrated
}
pred_contours = dict()

for model_name, clf in models.items():
    print(f'\n******* Fitting {model_name} *******\n')
    pred_contours[model_name] = fit_predict(clf)


# %%

def plot_contours(contours, ax):
    ax.contour(*param_grid.meshgrid(), 2 * contours, levels=[chi2.ppf(0.683, df=2),
                                                             chi2.ppf(0.9545, df=2),
                                                             chi2.ppf(0.9973, df=2)], colors=["w"])
    ax.contourf(*param_grid.meshgrid(), 2 * contours, vmin=0, vmax=30)
    ax.plot([true_alpha], [true_beta], "ro", markersize=8, label='True')
    ax.plot([alpha_mle], [beta_mle], "go", markersize=8, label='MLE')
    ax.set_xlim(*alpha_bounds)
    ax.set_ylim(*beta_bounds)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")


f, axarr = plt.subplots(2, sharex=True)
for i, contours in enumerate([exact_contours, pred_contours['Regular Calibrated']]):
    plot_contours(contours, ax=axarr[i])

axarr[0].legend(loc='upper center', fancybox=True, shadow=True)
plt.show()
