from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
from sklearn import clone
from sklearn.model_selection import StratifiedShuffleSplit

from active_learning_ratio_estimation.dataset import UnparameterizedRatioDataset
from active_learning_ratio_estimation.util import ideal_classifier_probs_from_simulator, negative_log_likelihood_ratio
from active_learning_ratio_estimation.model import UnparameterizedRatioModel, DenseClassifier, FlipoutClassifier
from active_learning_ratio_estimation.model.validation import get_calibration_metrics

from experiments.util import set_all_random_seeds, matplotlib_setup

quantities = ('y_pred', 'nllr')


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


def create_dataset(
        n_samples_per_theta: int,
        theta_0: float,
        theta_1: float
):
    ds = UnparameterizedRatioDataset.from_simulator(
        n_samples_per_theta=n_samples_per_theta,
        simulator_func=triple_mixture,
        theta_0=theta_0,
        theta_1=theta_1
    )
    return ds


def create_models(
        **hyperparams
):
    # regular, uncalibrated model
    regular_estimator = DenseClassifier(activation='tanh', **hyperparams)
    regular_uncalibrated = UnparameterizedRatioModel(
        estimator=regular_estimator,
        calibration_method=None,
        normalize_input=False
    )

    # bayesian, uncalibrated model
    bayesian_estimator = FlipoutClassifier(activation='relu', **hyperparams)
    bayesian_uncalibrated = UnparameterizedRatioModel(
        estimator=bayesian_estimator,
        calibration_method=None,
        normalize_input=False
    )

    # regular, calibrated model
    cv = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=1)
    regular_calibrated = UnparameterizedRatioModel(
        estimator=clone(regular_estimator),
        calibration_method='sigmoid',
        normalize_input=False,
        cv=cv
    )

    models = {
        'Regular Uncalibrated': regular_uncalibrated,
        'Bayesian Uncalibrated': bayesian_uncalibrated,
        'Regular Calibrated': regular_calibrated
    }
    return models


def fit_predict_models(
        models,
        x,
        dataset,
        verbose=False
):
    columns = pd.MultiIndex.from_product([quantities, models])
    results = pd.DataFrame(columns=columns, index=x)

    for model_name, model in models.items():
        if verbose:
            print(f'\n******* Fitting {model_name} *******\n')
        model.fit(dataset)
        results['y_pred', model_name] = model.predict_proba(x)[:, 1]
        results['nllr', model_name] = model.predict_negative_log_likelihood_ratio(x)

    theta_0, theta_1 = dataset.theta_0, dataset.theta_1
    results['y_pred', 'Ideal'] = ideal_classifier_probs_from_simulator(x, triple_mixture, theta_0, theta_1)
    results['nllr', 'Ideal'] = negative_log_likelihood_ratio(x, triple_mixture, theta_0, theta_1)

    return results


def calculate_mse(results):
    mses = pd.Series(dtype=float)
    y_preds = results['y_pred']

    for model_name in y_preds.columns:
        if model_name == 'Ideal':
            continue
        mses[model_name] = np.mean((y_preds[model_name] - y_preds['Ideal']) ** 2)

    return mses


def get_calibration_info(
        models,
        dataset,
        n_data
):
    calibration_curves, scores = get_calibration_metrics(
        ratio_models=models,
        dataset=dataset,
        n_data=n_data,
        n_bins=20
    )
    return calibration_curves, scores


def run_single_experiment(
        n_samples_per_theta: int,
        theta_0: float,
        theta_1: float,
        hyperparams: Dict,
        n_data: int
):
    ds = create_dataset(
        n_samples_per_theta=n_samples_per_theta,
        theta_0=theta_0,
        theta_1=theta_1
    )
    models = create_models(**hyperparams)
    x = np.linspace(-5, 5, int(1e4))
    results = fit_predict_models(models, x, ds)
    calibration_curves, scores = get_calibration_info(models, ds, n_data)
    return dict(results=results, scores=scores, calibration_curves=calibration_curves)


def run_experiments():
    set_all_random_seeds()
    matplotlib_setup()
