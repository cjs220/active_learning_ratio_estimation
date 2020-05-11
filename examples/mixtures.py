import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from dataset import RatioDataset
from util import ideal_classifier_probs, negative_log_likelihood_ratio
from model import FrequentistUnparameterisedRatioModel, BayesianUnparameterisedRatioModel


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


# Build dataset
theta_0 = 0.05
theta_1 = 0.00
n_samples_per_theta = int(1e5)

ds = RatioDataset(
    n_samples_per_theta=n_samples_per_theta,
    simulator_func=triple_mixture,
    theta_0_dist=theta_0,
    theta_1_dist=theta_1
)

# hyperparams
epochs = 10
patience = 2
lr = 1e-3
validation_split = 0.1

n_hidden = (10, 10)
activations = (
    'tanh',
    'relu'
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=patience),
    tf.keras.callbacks.TensorBoard()
]

x_dim = 1
num_samples = int(validation_split * 2 * n_samples_per_theta)

# Choose model types
model_types = [
    ('Regular', FrequentistUnparameterisedRatioModel),
    ('Bayesian', BayesianUnparameterisedRatioModel)
]

# Evaluate over [-5, 5]
x = np.linspace(-5, 5, int(1e4))
df = pd.DataFrame(index=x)
models = {}

for (name, model_cls), activation in zip(model_types, activations):
    # fit model
    clf = model_cls(x_dim=1,
                    num_samples=num_samples,
                    activation=activation,
                    n_hidden=n_hidden)
    clf.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    clf.fit_dataset(ds,
                    epochs=epochs,
                    validation_split=validation_split,
                    callbacks=callbacks,
                    verbose=2)
    models[name] = clf

    # predict over x
    y_pred = clf.predict(x, theta_0, theta_1)
    lr_estimate = clf.predict_likelihood_ratio(x, theta_0, theta_1)
    nllr = -np.log(lr_estimate)
    df[f'y_pred ({name})'] = y_pred.squeeze()
    df[f'NLLR ({name})'] = nllr.squeeze()

# Find ideal classifier probabilities, and true likelihood ratio
y_pred_ideal = ideal_classifier_probs(x, triple_mixture, theta_0, theta_1)
df['y_pred (Ideal)'] = y_pred_ideal
df['NLLR (True)'] = negative_log_likelihood_ratio(x, triple_mixture, theta_0, theta_1)

# Plot results
f, axarr = plt.subplots(2, sharex=True)

for i, variable in enumerate(['y_pred', 'NLLR']):
    cols = list(filter(lambda x: variable in x, df.columns))
    for col in cols:
        alpha = 0.5 if 'Bayesian' in col else 1
        df[col].plot(ax=axarr[i], alpha=alpha)
    axarr[i].legend()
    plt.xlim([-1, 3])

# Look at calibration
f, axarr = plt.subplots(2, sharex=True)
for y in (0, 1):
    filter_ds = ds.filter(ds.y == y)
    df_calib = pd.DataFrame()
    for name, clf in models.items():
        pred = clf.predict_dataset(filter_ds)
        col_name = f'{name} (y={y})'
        df_calib[col_name] = pred.squeeze()
        df_calib[col_name].plot.kde(ax=axarr[y], label=col_name)
        axarr[y].legend()
    plt.legend()
    plt.xlim([0.4, 0.6])

plt.show()
