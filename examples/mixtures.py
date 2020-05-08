import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from dataset import RatioDataset
from dist import triple_mixture, ideal_classifier_probs, negative_log_likelihood_ratio
from model import FrequentistUnparameterisedRatioModel, BayesianUnparameterisedRatioModel

# Build dataset
simulator_func = triple_mixture
theta_0 = 0.05
theta_1 = 0
n_samples_per_theta = int(1e5)

ds = RatioDataset(
    n_thetas=1,
    n_samples_per_theta=n_samples_per_theta,
    simulator_func=simulator_func,
    theta_0_dist=theta_0,
    theta_1_dist=theta_1
)

# hyperparams
epochs = 10
patience = 10
lr = 1e-3
validation_split = 0.1
callbacks = [
    tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=patience),
    tf.keras.callbacks.TensorBoard()
]

x_dim = len(ds.x)
num_samples = int(validation_split * 2 * n_samples_per_theta)


# Choose model types
model_types = [
    ('Regular', FrequentistUnparameterisedRatioModel),
    ('Bayesian', BayesianUnparameterisedRatioModel)
]

# Evaluate over [-5, 5]
x = np.linspace(-5, 5, int(1e4))
df = pd.DataFrame(index=x)
models = []

for name, model_cls in model_types:
    # fit model
    clf = model_cls(x_dim=1, num_samples=num_samples, activation='tanh')
    clf.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
    )
    clf.fit_dataset(ds,
                    epochs=epochs,
                    validation_split=validation_split,
                    callbacks=callbacks)
    models.append(clf)

    # predict over x
    y_pred = clf.predict(x, theta_0, theta_1)
    lr_estimate = clf.predict_likelihood_ratio(x, theta_0, theta_1)
    nllr = -np.log(lr_estimate)
    df[f'y_pred ({name})'] = y_pred.squeeze()
    df[f'NLLR ({name})'] = nllr.squeeze()

# Find ideal classifier probabilities, and true likelihood ratio
y_pred_ideal = ideal_classifier_probs(x, simulator_func, theta_0, theta_1)
df['y_pred (Ideal)'] = y_pred_ideal
df['NLLR (True)'] = negative_log_likelihood_ratio(x, simulator_func, theta_0, theta_1)

f, axarr = plt.subplots(2)
for i, variable in enumerate(['y_pred', 'NLLR']):
    cols = list(filter(lambda x: variable in x, df.columns))
    df[cols].plot(ax=axarr[i])
plt.show()
