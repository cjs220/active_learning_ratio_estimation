from typing import List, Iterable, Dict, Callable
import os
import random

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


def matplotlib_setup(size=24, use_tex=True):
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
    config = config or {}

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

    config_path = os.path.join(dir_path, 'config.txt')
    with open(config_path, 'w+') as outfile:
        outfile.write(str(config))


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


if __name__ == '__main__':
    frames = {f'frame_{i + 1}': pd.DataFrame(np.random.rand(10, 3), columns=list('abc')) for i in range(5)}
    figures = dict()

    for i in range(3):
        fig, ax = plt.subplots()
        ax.plot(np.linspace(0, 1, 101), np.linspace(0, 1, 101) ** (i + 1))
        figures[f'figure_{i + 1}'] = fig

    config = dict(x=5, y=0, z=6)
    save_results('test', figures=figures, frames=frames, config=config)
