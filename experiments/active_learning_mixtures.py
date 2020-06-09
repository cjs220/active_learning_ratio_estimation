from typing import Sequence, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from active_learning_ratio_estimation.active_learning import ActiveLearner
from active_learning_ratio_estimation.dataset import ParamGrid, ParamIterator, SinglyParameterizedRatioDataset
from active_learning_ratio_estimation.model import FlipoutClassifier, SinglyParameterizedRatioModel

from experiments.mixtures import triple_mixture
from experiments.util import set_all_random_seeds, matplotlib_setup, run_parallel_experiments, save_results

quantities = ('val_loss', 'val_accuracy', 'test_mse')
TRAIN_THETA_BOUNDS = (0, 1)


def get_active_learner(
        acquisition_function,
        theta_0,
        theta_1_iterator,
        n_samples_per_theta,
        param_grid,
        test_dataset,
        **hyperparams
):
    estimator = FlipoutClassifier(**hyperparams)
    ratio_model = SinglyParameterizedRatioModel(estimator=estimator)

    active_learner = ActiveLearner(
        simulator_func=triple_mixture,
        theta_0=theta_0,
        theta_1_iterator=theta_1_iterator,
        n_samples_per_theta=n_samples_per_theta,
        ratio_model=ratio_model,
        total_param_grid=param_grid,
        test_dataset=test_dataset,
        acquisition_function=acquisition_function,
        ucb_kappa=0.0,
        validation_mode=False,
    )
    return active_learner


def collect_results(active_learners):
    index = pd.MultiIndex.from_product([quantities, active_learners.keys()])
    results = pd.DataFrame(columns=index)
    trialed_thetas = pd.DataFrame(columns=active_learners.keys())

    for acquisition_function, learner in active_learners.items():
        for quantity in quantities[:-1]:
            results[quantity, acquisition_function] = learner.train_history[quantity].values

        trialed_thetas[acquisition_function] = learner.trialed_thetas.squeeze()

        if learner.test_dataset is not None:
            results['test_mse', acquisition_function] = learner.test_history.values

    return results, trialed_thetas


def plot_all_results(aggregated_results):
    means = aggregated_results.mean(axis=0, level=1)
    stds = aggregated_results.std(axis=0, level=1, ddof=1)
    n = len(aggregated_results.columns.levels[0])
    stderrs = stds / np.sqrt(n)

    fig, axarr = plt.subplots(len(quantities), figsize=(15, 7), sharex=True)
    colours = ('r', 'b', 'g')
    for ax, quantity_name in zip(axarr, quantities):
        quantity_means = means[quantity_name]
        quantity_stderrs = stderrs[quantity_name]
        for i, af in enumerate(quantity_means.columns):
            mean = quantity_means[af]
            mean.plot(ax=ax, marker='o', color=colours[i])
            stderr = quantity_stderrs[af]
            ax.fill_between(
                x=mean.index.values,
                y1=mean.values + stderr.values,
                y2=mean.values - stderr.values,
                alpha=0.3,
                color=colours[i]
            )
            ax.set_title(quantity_name)
    axarr[0].legend()
    return fig


def run_single_experiment(
        acquisition_functions: Sequence[str],
        n_iter: int,
        theta_0: float,
        num_grid: int,
        n_samples_per_theta: int,
        n_train_init: int,
        n_theta_test: int,
        hyperparams: Dict,
        verbose: bool = False
):
    param_grid = ParamGrid(bounds=[TRAIN_THETA_BOUNDS], num=num_grid)  # all possible parameter points
    theta_1_iterator = ParamGrid(bounds=[TRAIN_THETA_BOUNDS], num=n_train_init)  # initial parameter points in dataset

    # test dataset
    if n_theta_test > 0:
        test_iterator = ParamIterator([np.random.rand(1) for _ in range(n_theta_test)])
        test_dataset = SinglyParameterizedRatioDataset.from_simulator(
            simulator_func=triple_mixture,
            theta_0=theta_0,
            theta_1_iterator=test_iterator,
            n_samples_per_theta=n_samples_per_theta,
            include_log_probs=True
        )
    else:
        test_dataset = None

    active_learners = dict()
    for acquisition_function in acquisition_functions:
        if verbose:
            print(f'\nFitting with {acquisition_function} acquisition function')
        active_learner = get_active_learner(
            acquisition_function=acquisition_function,
            theta_0=theta_0,
            theta_1_iterator=theta_1_iterator,
            n_samples_per_theta=n_samples_per_theta,
            param_grid=param_grid,
            test_dataset=test_dataset,
            **hyperparams
        )
        active_learner.fit(n_iter, verbose=verbose)
        active_learners[acquisition_function] = active_learner
    results, trialed_thetas = collect_results(active_learners)
    return results, trialed_thetas


def run_all_experiments(n_experiments, n_jobs, **run_kwargs):
    set_all_random_seeds()
    matplotlib_setup(use_tex=False)
    experiment_outcomes = run_parallel_experiments(
        experiment_func=run_single_experiment,
        n_experiments=n_experiments,
        n_jobs=n_jobs,
        **run_kwargs
    )
    all_results, all_trialed_thetas = zip(*experiment_outcomes)
    aggregated_results = pd.concat(all_results, axis=0, keys=range(len(all_results)))
    all_trialed_thetas = pd.concat(all_trialed_thetas, axis=1, keys=range(len(all_results)))
    results_plot = plot_all_results(aggregated_results)
    save_results(
        'active_learning_mixtures',
        figures=dict(results=results_plot),
        frames=dict(
            results=aggregated_results,
            thetas=all_trialed_thetas
        )
    )


if __name__ == '__main__':
    run_kwargs = dict(
        acquisition_functions=('std', 'random'),
        n_iter=15,
        theta_0=0.05,
        num_grid=101,
        n_samples_per_theta=int(1e3),
        n_train_init=3,
        n_theta_test=10,
        verbose=True,
        hyperparams=dict(
            n_hidden=(15, 15),
            epochs=5,
            patience=0,
            validation_split=0.1,
            verbose=0,
        ),
    )
    run_all_experiments(n_experiments=4, n_jobs=-1, **run_kwargs)
    # run_single_experiment(**run_kwargs)
