from typing import Sequence, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import trim_mean
from sklearn import clone

from active_learning_ratio_estimation.model.ratio_model import calibrated_param_scan, param_scan, exact_param_scan
from carl.learning import CalibratedClassifierCV
import tensorflow_probability as tfp

from experiments.util import matplotlib_setup, set_all_random_seeds, run_parallel_experiments, \
    plot_line_graph_with_errors, save_results

tfd = tfp.distributions

from active_learning_ratio_estimation.dataset import SinglyParameterizedRatioDataset, ParamGrid, \
    DistributionParamIterator, ParamIterator
from active_learning_ratio_estimation.model import DenseClassifier, SinglyParameterizedRatioModel, FlipoutClassifier

from experiments.mixtures import triple_mixture


def create_models(
        theta_0: float,
        hyperparams: Dict,
) -> Dict[str, SinglyParameterizedRatioModel]:
    # regular, uncalibrated model
    regular_estimator = DenseClassifier(activation='tanh', **hyperparams)
    regular_uncalibrated = SinglyParameterizedRatioModel(theta_0=theta_0, clf=regular_estimator)

    # bayesian, uncalibrated model
    bayesian_estimator = FlipoutClassifier(activation='relu', **hyperparams)
    bayesian_uncalibrated = SinglyParameterizedRatioModel(theta_0=theta_0, clf=bayesian_estimator)

    models = {
        'Regular': regular_uncalibrated,
        'Bayesian': bayesian_uncalibrated,
    }
    return models


def fit_and_predict_models(
        models: Dict[str, SinglyParameterizedRatioModel],
        train_dataset: SinglyParameterizedRatioDataset,
        test_dataset: SinglyParameterizedRatioDataset,
) -> pd.DataFrame:
    predictions = pd.DataFrame(dict(
        theta=test_dataset.theta_1s.squeeze(),
        x=test_dataset.x.squeeze(),
        Exact=(test_dataset.log_prob_1 - test_dataset.log_prob_0).squeeze()
    ))
    for model_name, model in models.items():
        model.fit(train_dataset.x, train_dataset.theta_1s, train_dataset.y)
        logr = model.predict(test_dataset.x, test_dataset.theta_1s, log=True)
        predictions[model_name] = logr
    return predictions


def calibrated_predict(
        model: SinglyParameterizedRatioModel,
        test_dataset: SinglyParameterizedRatioDataset,
        n_calibration: int,
        calibration_params: Dict,
) -> np.ndarray:
    predictions = np.array([])
    for theta in np.unique(test_dataset.theta_1s, axis=0):
        dataset_slice = test_dataset[np.all(test_dataset.theta_1s == theta, axis=1)]
        new_predictions = model.calibrated_predict(
            X=dataset_slice.x,
            theta=theta,
            n_samples_per_theta=n_calibration,
            simulator_func=triple_mixture,
            calibration_params=calibration_params,
            log=True,
        )
        predictions = np.append(predictions, new_predictions)
    return predictions


def run_param_scan(
        model: SinglyParameterizedRatioModel,
        X_true: np.ndarray,
        param_grid: ParamGrid,
        n_calibration: int = None,
        calibration_params: Dict = None
):
    if calibration_params is not None:
        return calibrated_param_scan(
            model=model,
            X_true=X_true,
            param_grid=param_grid,
            simulator_func=triple_mixture,
            n_samples_per_theta=n_calibration,
            calibration_params=calibration_params
        )
    else:
        return param_scan(
            model=model,
            X_true=X_true,
            param_grid=param_grid,
            theta_batch_size=10,
        )


def plot_mse_vs_theta(mse: pd.DataFrame, trimmed_mse: pd.DataFrame) -> Figure:
    fig, axarr = plt.subplots(2, sharex=True)

    def _plot_mse(df, ax):
        mean = df.mean(axis=0, level=1)
        stderr = df.sem(axis=0, level=1)
        plot_line_graph_with_errors(mean, stderr, ax=ax)

    for ax, df, title in zip(axarr, [mse, trimmed_mse], ['MSE', 'Trimmed MSE']):
        _plot_mse(df, ax)
        ax.set_title(title)

    axarr[1].set_xlabel(r'$\theta$')
    return fig


def plot_total_mse(mse: pd.DataFrame, trimmed_mse: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots()
    average_errors = {'MSE': mse.mean(level=0), 'Trimmed MSE': trimmed_mse.mean(level=0)}

    mean_average_errors = pd.concat(
        [average_err.mean(axis=0).rename(err_name) for err_name, average_err in average_errors.items()],
        axis=1
    ).T
    stderr_average_errors = pd.concat(
        [average_err.sem(axis=0).rename(err_name) for err_name, average_err in average_errors.items()],
        axis=1
    ).T

    mean_average_errors.plot.bar(
        ax=ax,
        yerr=stderr_average_errors,
        alpha=0.5,
        capsize=10,
        rot=0,
    )
    return fig


def plot_test_stat(test_stat: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots()
    mean = test_stat.mean(axis=0, level=1)
    stderr = test_stat.sem(axis=0, level=1)
    ax = plot_line_graph_with_errors(mean=mean, stderr=stderr, ax=ax)
    ax.set(xlabel=r'$\theta$', title=r'$-2 \, \log \frac{L(\theta)}{L(\theta_{MLE})}$')
    return fig


def plot_mle_distributions(mle: pd.DataFrame, theta_true: float) -> Figure:
    mle = mle.astype(np.float32)
    fig, axarr = plt.subplots(mle.shape[1] - 1, sharex=True)
    non_exact_cols = filter(lambda x: x != 'Exact', mle.columns)
    for ax, col in zip(axarr, non_exact_cols):
        for model in ('Exact', col):
            mle[model].plot.hist(ax=ax, alpha=0.3, density=True)
        ax.set_prop_cycle(None)
        for model in ('Exact', col):
            mle[model].plot.kde(ax=ax, label=None)
        ax.axvline(x=theta_true, label='True', color='k')
        ax.legend()
    return fig


def run_single_experiment(
        theta_0: float,
        theta_bounds_train: Sequence[float],
        n_samples_per_theta_train: int,
        n_thetas_train: int,
        theta_test_values: np.array,
        n_samples_per_theta_test: int,
        hyperparams: Dict,
        calibration_params: Dict,
        n_calibration: int,
        theta_true: float,
        n_true: int
) -> Sequence[pd.DataFrame]:
    train_grid = ParamGrid(bounds=[theta_bounds_train], num=n_thetas_train)
    train_ds = SinglyParameterizedRatioDataset.from_simulator(
        simulator_func=triple_mixture,
        theta_0=theta_0,
        n_samples_per_theta=n_samples_per_theta_train,
        theta_1_iterator=train_grid,
        include_log_probs=False
    )

    test_iterator = ParamIterator([theta for theta in theta_test_values])
    test_ds = SinglyParameterizedRatioDataset.from_simulator(
        simulator_func=triple_mixture,
        theta_0=theta_0,
        n_samples_per_theta=n_samples_per_theta_test,
        theta_1_iterator=test_iterator,
        include_log_probs=True
    )
    test_ds = test_ds[test_ds.y == 1]

    models = create_models(theta_0=theta_0, hyperparams=hyperparams)

    predictions = fit_and_predict_models(
        models=models,
        train_dataset=train_ds,
        test_dataset=test_ds
    )
    predictions['Calibrated'] = calibrated_predict(
        model=models['Regular'],
        test_dataset=test_ds,
        n_calibration=n_calibration,
        calibration_params=calibration_params
    )
    predictions = predictions.sort_values(by='theta').set_index(['theta', 'x'])
    squared_errors = ((predictions.subtract(predictions['Exact'], axis=0)) ** 2).drop('Exact', axis=1)
    mse = squared_errors.mean(level=0)
    trimmed_mse = pd.concat(
        [squared_errors[col].groupby('theta').apply(trim_mean, 0.05) for col in squared_errors.columns],
        axis=1
    )

    X_true = triple_mixture(theta_true).sample(n_true).numpy()

    param_scan_results = {
        model_name: run_param_scan(model=model, X_true=X_true, param_grid=train_grid)
        for model_name, model in models.items()
    }
    param_scan_results['Calibrated'] = run_param_scan(
        model=models['Regular'],
        X_true=X_true,
        param_grid=train_grid,
        n_calibration=n_calibration,
        calibration_params=calibration_params
    )
    nllr, mle = zip(*param_scan_results.values())
    nllr = pd.DataFrame(dict(zip(param_scan_results.keys(), nllr)), index=train_grid.array.squeeze())
    mle = pd.Series(dict(zip(param_scan_results.keys(), np.array(mle).squeeze())))
    nllr_exact, mle_exact = exact_param_scan(
        simulator_func=triple_mixture,
        X_true=X_true,
        param_grid=train_grid,
        theta_0=theta_0
    )
    nllr['Exact'] = nllr_exact
    mle['Exact'] = mle_exact.squeeze()
    test_stat = 2*(nllr - nllr.min())

    return mse, trimmed_mse, test_stat, mle


def run_all_experiments(**run_kwargs):
    matplotlib_setup()
    set_all_random_seeds()

    n_experiments = 20
    n_theta_test = 101
    theta_test_values = np.random.uniform(*run_kwargs['theta_bounds_train'], size=n_theta_test).reshape(-1, 1)

    # run_single_experiment(theta_test_values=theta_test_values, **run_kwargs)

    results = run_parallel_experiments(
        experiment_func=run_single_experiment,
        n_experiments=n_experiments,
        theta_test_values=theta_test_values,
        **run_kwargs
    )
    mse, trimmed_mse, test_stat, mle = zip(*results)

    mse = pd.concat(mse, axis=0, keys=range(n_experiments))
    trimmed_mse = pd.concat(trimmed_mse, axis=0, keys=range(n_experiments))
    mse_vs_theta_fig = plot_mse_vs_theta(mse, trimmed_mse)
    total_mse_fig = plot_total_mse(mse, trimmed_mse)

    test_stat = pd.concat(test_stat, axis=0, keys=range(n_experiments))
    exclusion_fig = plot_test_stat(test_stat)

    mle = pd.concat(mle, axis=1).T
    mle_fig = plot_mle_distributions(mle=mle, theta_true=run_kwargs['theta_true'])

    save_results(
        experiment_name=__file__.split('/')[-1].split('.py')[0],
        figures=dict(mse_vs_theta=mse_vs_theta_fig, total_mse=total_mse_fig, exclusion=exclusion_fig, mle=mle_fig),
        frames=dict(mse=mse, trimmed_mse=trimmed_mse),
        config={'n_experiments': n_experiments, 'theta_test_values': theta_test_values, **run_kwargs}
    )


if __name__ == '__main__':
    matplotlib_setup()
    set_all_random_seeds()

    N = int(1e3)

    run_kwargs = dict(
        theta_0=0.00,
        theta_bounds_train=(0, 0.12),
        n_samples_per_theta_train=N,
        n_thetas_train=101,
        n_samples_per_theta_test=N,
        hyperparams=dict(
            n_hidden=(10, 10),
            epochs=100,
            patience=5,
            validation_split=0.1,
            verbose=2,
            fit_batch_size=32,
            predict_batch_size=32
        ),
        calibration_params=dict(
            method='histogram',
            bins=20,
            interpolation='slinear'
        ),
        n_calibration=N,
        theta_true=0.05,
        n_true=N
    )

    run_all_experiments(**run_kwargs)
