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

def main():
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
    epochs = 50
    patience = 5
    lr = 1e-3
    validation_split = 0.1
    n_hidden = (10, 10)
    num_samples = int(validation_split * 2 * n_samples_per_theta)

    compile_kwargs = dict(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    fit_kwargs = dict(
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=patience),
            tf.keras.callbacks.TensorBoard()
        ],
        verbose=2
    )


    # Choose model types
    model_types = [
        ('Regular', FrequentistUnparameterisedRatioModel, 'tanh'),
        ('Bayesian', BayesianUnparameterisedRatioModel, 'relu')
    ]

    # Evaluate over [-5, 5]
    x = np.linspace(-5, 5, int(1e4))
    df = pd.DataFrame(index=x)
    models = {}

    for name, model_cls, activation in model_types:
        for calibrated in (True, False):
            calibration = 'isotonic' if calibrated else None

            # fit model
            clf = model_cls(x_dim=1,
                            theta_dim=1,
                            num_samples=num_samples,
                            activation=activation,
                            n_hidden=n_hidden,
                            calibration=calibration,
                            compile_kwargs=compile_kwargs,
                            fit_kwargs=fit_kwargs)
            clf.fit_dataset(ds)
            models[name] = clf

            # predict over x
            y_pred = clf.predict_proba(x, theta_0, theta_1)[:, 1]
            lr_estimate = clf.predict_likelihood_ratio(x, theta_0, theta_1)
            nllr = -np.log(lr_estimate)
            calibration_label = 'Calibrated' if calibrated else 'Uncalibrated'
            label = f'({name}, {calibration_label})'
            df[f'y_pred {label}'] = y_pred.squeeze()
            df[f'NLLR {label}'] = nllr.squeeze()

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
            pred = clf.predict_proba_dataset(filter_ds)[:, 1]
            col_name = f'{name} (y={y})'
            df_calib[col_name] = pred.squeeze()
            df_calib[col_name].plot.kde(ax=axarr[y], label=col_name)
            axarr[y].legend()
    plt.legend()
    plt.xlim([0.4, 0.6])

    plt.show()


if __name__ == '__main__':
    main()
