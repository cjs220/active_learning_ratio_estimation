from typing import List, Iterable, Dict, Callable
import os
import random
import pprint

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.figure import Figure
from pandas.core.generic import NDFrame
import matplotlib.pyplot as plt
import tensorflow as tf


def disable_tensorflowgpu():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def set_all_random_seeds(seed=0):
    np.random.seed(seed)
    tf.random.set_seed(0)
    random.seed(0)


def matplotlib_setup(size=24, use_tex=False):
    params = {
        'legend.fontsize': size * 0.75,
        'figure.figsize': (10, 5),
        'axes.labelsize': size,
        'axes.titlesize': size,
        'xtick.labelsize': size * 0.75,
        'ytick.labelsize': size * 0.75,
        'font.family': 'sans-serif',
        'axes.titlepad': 12.5,
        'text.usetex': use_tex
    }
    plt.rcParams.update(params)


def save_results(
        experiment_name: str,
        root_folder: str = None,
        figures: Dict[str, Figure] = None,
        frames: Dict[str, NDFrame] = None,
        config: Dict = None,
):
    root_folder = root_folder or 'results'
    figures = figures or {}
    frames = frames or {}

    # make directory
    time_str = pd.Timestamp.now().strftime('%Y-%m-%d_%H%M')
    dir_name = f'{time_str}_{experiment_name}'
    dir_path = os.path.join(root_folder, dir_name)
    os.mkdir(dir_path)

    for fig_name, fig in figures.items():
        fig_path = os.path.join(dir_path, fig_name)
        fig.tight_layout()
        fig.savefig(fig_path + '.svg')

    for frame_name, frame in frames.items():
        frame_path = os.path.join(dir_path, frame_name)
        frame.to_csv(frame_path + '.csv')

    if config:
        config_path = os.path.join(dir_path, 'config.txt')
        with open(config_path, 'w+') as outfile:
            outfile.write(pprint.pformat(config))


def run_parallel_experiments(
        experiment_func: Callable,
        n_experiments: int,
        n_jobs: int = -2,
        verbose: int = 10,
        **experiment_func_kwargs
):
    return Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(experiment_func)(**experiment_func_kwargs) for _ in range(n_experiments)
    )


def plot_line_graph_with_errors(mean: pd.DataFrame, stderr: pd.DataFrame, ax=None, alpha=0.3, **kwargs):
    ax = mean.plot(ax=ax)
    ax.set_prop_cycle(None)
    for col in stderr.columns:
        ax.fill_between(
            mean.index.values,
            mean[col].values - stderr[col].values,
            mean[col].values + stderr[col].values,
            alpha=alpha,
            **kwargs
        )
    return ax
