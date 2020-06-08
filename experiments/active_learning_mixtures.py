import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from active_learning_ratio_estimation.active_learning import ActiveLearner
from active_learning_ratio_estimation.dataset import ParamGrid
from active_learning_ratio_estimation.model import FlipoutClassifier, SinglyParameterizedRatioModel

from experiments.mixtures import triple_mixture
from experiments.util import set_all_random_seeds, matplotlib_setup, run_parallel_experiments, save_results

quantities = ('val_loss', 'val_accuracy', 'test_mse')


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
    results_df = pd.DataFrame(columns=index)

    for acquisition_function, learner in active_learners.items():
        for quantity in quantities[:-1]:
            results_df[quantity, acquisition_function] = learner.train_history[quantity].values

        if learner.test_dataset is not None:
            results_df['test_mse', acquisition_function] = learner.test_history.values

    return results_df


def run_single_experiment():
    n_iter = 5
    theta_bounds = (0, 1)
    theta_0 = 0.05
    num_grid = 101
    n_samples_per_theta = int(1e3)
    param_grid = ParamGrid(bounds=[theta_bounds], num=num_grid)  # all possible parameter points
    theta_1_iterator = ParamGrid(bounds=[theta_bounds], num=3)  # initial parameter points in dataset
    hyperparams = dict(
        n_hidden=(15, 15),
        epochs=5,
        patience=0,
        validation_split=0.1,
        verbose=0,
    )

    active_learners = dict()
    for acquisition_function in ('std_regressor', 'random'):
        active_learner = get_active_learner(
            acquisition_function=acquisition_function,
            theta_0=theta_0,
            theta_1_iterator=theta_1_iterator,
            n_samples_per_theta=n_samples_per_theta,
            param_grid=param_grid,
            test_dataset=test_dataset,
            **hyperparams
        )
        active_learner.fit(n_iter, verbose=False)
        active_learners[acquisition_function] = active_learner
    results = collect_results(active_learners)
    return results


def plot_all_results(aggregated_results):
    means = aggregated_results.mean(axis=0, level=0)
    stds = aggregated_results.std(axis=0, level=0, ddof=1)
    n = len(aggregated_results.columns.levels[0])
    stderrs = stds / np.sqrt(n)

    fig, axarr = plt.subplots(2, figsize=(15, 7), sharex=True)
    colours = ('r', 'b')
    for ax, quantity_name in zip(axarr, quantities[:-1]):
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


def run_all_experiments(n_experiments, n_jobs):
    set_all_random_seeds()
    matplotlib_setup(use_tex=False)
    all_results = run_parallel_experiments(
        experiment_func=run_single_experiment,
        n_experiments=n_experiments,
        n_jobs=n_jobs
    )
    aggregated_results = pd.concat(all_results, axis=0, keys=range(len(all_results)))
    results_plot = plot_all_results(aggregated_results)
    save_results(
        'active_learning_mixtures',
        figures=dict(results=results_plot),
        frames=dict(results=aggregated_results)
    )


if __name__ == '__main__':
    run_all_experiments(n_experiments=3, n_jobs=3)

