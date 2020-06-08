from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib.figure import Figure
from pandas.core.generic import NDFrame

tfd = tfp.distributions
from sklearn import clone
from sklearn.model_selection import StratifiedShuffleSplit

from active_learning_ratio_estimation.dataset import UnparameterizedRatioDataset
from active_learning_ratio_estimation.util import ideal_classifier_probs_from_simulator, negative_log_likelihood_ratio
from active_learning_ratio_estimation.model import UnparameterizedRatioModel, DenseClassifier, FlipoutClassifier
from active_learning_ratio_estimation.model.validation import get_calibration_metrics

from experiments.util import set_all_random_seeds, matplotlib_setup, run_parallel_experiments, save_results

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


def create_models(**hyperparams):
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
        calibration_method='histogram',
        normalize_input=False,
        cv=cv,
        bins=15,
        interpolation='quadratic'
    )

    models = {
        'Regular Uncalibrated': regular_uncalibrated,
        'Bayesian Uncalibrated': bayesian_uncalibrated,
        'Regular Calibrated': regular_calibrated
    }
    return models


def fit_predict_models(
        models,
        dataset,
        x,
        verbose=False
):
    columns = pd.MultiIndex.from_product([quantities, models])
    predictions = pd.DataFrame(columns=columns, index=x)

    for model_name, model in models.items():
        if verbose:
            print(f'\n******* Fitting {model_name} *******\n')
        model.fit(dataset)
        predictions['y_pred', model_name] = model.predict_proba(x)[:, 1]
        predictions['nllr', model_name] = model.predict_negative_log_likelihood_ratio(x)

    theta_0, theta_1 = dataset.theta_0, dataset.theta_1
    predictions['y_pred', 'Ideal'] = ideal_classifier_probs_from_simulator(x, triple_mixture, theta_0, theta_1)
    predictions['nllr', 'Ideal'] = negative_log_likelihood_ratio(x, triple_mixture, theta_0, theta_1)

    return predictions


def calculate_mse(predictions):
    mses = pd.Series(dtype=float)
    y_preds = predictions['y_pred']

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


# noinspection PyTypeChecker
def _aggreate_experiment_results(results: List[Dict[str, NDFrame]]):
    aggregated_predictions = pd.concat([res['predictions'] for res in results], axis=1, keys=range(len(results)))
    aggregated_scores = pd.concat([res['scores'] for res in results], axis=1)
    return dict(predictions=aggregated_predictions, scores=aggregated_scores)


def plot_scores(aggregate_scores: pd.DataFrame) -> Figure:
    means = aggregate_scores.mean(axis=1)
    stds = aggregate_scores.std(axis=1, ddof=1)
    n = aggregate_scores.shape[1]
    stderrs = stds / np.sqrt(n)
    fig, axarr = plt.subplots(2, 2, figsize=(15, 7), sharex=True)
    for ax, score_name in zip(np.ravel(axarr), means.index.levels[0]):
        mean = means[score_name]
        stderr = stderrs[score_name]
        if score_name in ('Brier Score', 'F1 (Micro)'):
            ylim = (mean - stderr).min(), (mean + stderr).max()
        else:
            ylim = None
        mean.plot.bar(
            ax=ax,
            yerr=stderr,
            ylim=ylim,
            alpha=0.5,
            capsize=10,
            rot=30,
            title=score_name
        )
    return fig


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
    predictions = fit_predict_models(models=models, x=x, dataset=ds)
    mses = calculate_mse(predictions)
    calibration_curves, scores = get_calibration_info(models, ds, n_data)
    for model_name, mse in mses.iteritems():
        scores['MSE', model_name] = mse
    return dict(predictions=predictions, scores=scores, calibration_curves=calibration_curves)


def run_experiments(n_experiments: int, **run_kwargs):
    set_all_random_seeds()
    matplotlib_setup()
    results = run_parallel_experiments(run_single_experiment, n_experiments, **run_kwargs)
    aggregated_results = _aggreate_experiment_results(results)
    scores_plot = plot_scores(aggregated_results['scores'])
    save_results(
        experiment_name='mixtures',
        figures=dict(scores=scores_plot),
        frames=aggregated_results,
        config={'n_experiments': n_experiments, **run_kwargs}
    )


if __name__ == '__main__':
    n_experiments = 4
    run_kwargs = dict(
        n_samples_per_theta=int(1e5),
        theta_0=0.05,
        theta_1=0.00,
        hyperparams=dict(
            n_hidden=(10, 10),
            epochs=20,
            patience=2,
            validation_split=0.1,
            verbose=False,
        ),
        n_data=int(1e4)
    )
    run_experiments(n_experiments, **run_kwargs)
