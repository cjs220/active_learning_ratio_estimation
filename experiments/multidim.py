from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.stats import chi2
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
from sklearn.datasets import make_sparse_spd_matrix

from active_learning_ratio_estimation.model import SinglyParameterizedRatioModel, DenseClassifier, FlipoutClassifier
from active_learning_ratio_estimation.dataset import ParamGrid, SinglyParameterizedRatioDataset

from experiments.util import set_all_random_seeds, matplotlib_setup, save_results

ALPHA_TRUE = 1
BETA_TRUE = -1
TRUE_DATASET_SIZE = 500


class MultiDimToyModel(tfd.TransformedDistribution):

    def __init__(self, alpha, beta):
        self.alpha = tf.cast(alpha, tf.float32)
        self.beta = tf.cast(beta, tf.float32)

        # compose linear transform
        R = make_sparse_spd_matrix(5, alpha=0.5, random_state=7).astype(np.float32)
        self.R = R
        transform = tf.linalg.LinearOperatorFullMatrix(R)
        bijector = tfp.bijectors.AffineLinearOperator(scale=transform)

        super().__init__(distribution=self.z_distribution, bijector=bijector)

    @property
    def z_distribution(self):
        z_distribution = tfd.Blockwise([
            tfd.Normal(loc=self.alpha, scale=1),  # z1
            tfd.Normal(loc=self.beta, scale=3),  # z2
            tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=[0.5, 0.5]),
                components_distribution=tfd.Normal(
                    loc=[-2, 2],
                    scale=[1, 0.5]
                )
            ),  # z3
            tfd.Exponential(3),  # z4
            tfd.Exponential(0.5),  # z5
        ])
        return z_distribution


def get_X_true():
    p_true = MultiDimToyModel(alpha=ALPHA_TRUE, beta=BETA_TRUE)
    return p_true.sample(TRUE_DATASET_SIZE)


def create_dataset(
        theta_0: np.ndarray,
        n_samples_per_theta: int,
        n_theta_train: int
):
    train_grid = ParamGrid(bounds=[(-3, 3), (-3, 3)], num=n_theta_train)
    ds = SinglyParameterizedRatioDataset.from_simulator(
        simulator_func=MultiDimToyModel,
        theta_0=theta_0,
        n_samples_per_theta=n_samples_per_theta,
        theta_1_iterator=train_grid
    )
    return ds


def create_models(**hyperparams):
    # regular model
    regular_estimator = DenseClassifier(activation='tanh', **hyperparams)
    regular_model = SinglyParameterizedRatioModel(estimator=regular_estimator, calibration_method=None)

    # bayesian model
    bayesian_estimator = FlipoutClassifier(activation='relu', **hyperparams)
    bayesian_model = SinglyParameterizedRatioModel(estimator=bayesian_estimator, calibration_method=None)

    return dict(Regular=regular_model, Bayesian=bayesian_model)


def get_predict_grid(n_theta_pred: int):
    alpha_bounds = (0.75, 1.25)
    beta_bounds = (-2, 0)
    return ParamGrid(bounds=[alpha_bounds, beta_bounds], num=n_theta_pred)


def fit_predict_models(
        models: Dict[str, SinglyParameterizedRatioModel],
        dataset: SinglyParameterizedRatioDataset,
        X_true: np.ndarray,
        predict_grid: ParamGrid,
        n_calibration_points_per_theta: int
):
    predictions = dict()
    for model_name, model in models.items():
        model.fit(dataset)
        contours, mle = model.nllr_param_scan(
            x=X_true,
            param_grid=predict_grid,
            verbose=True
        )

        predictions[model_name] = dict(Contours=contours, MLE=mle)

        if model_name == 'Regular':
            calibrated_contours, calibrated_mle = model.nllr_param_scan_with_calibration(
                x=X_true,
                param_grid=predict_grid,
                simulator_func=MultiDimToyModel,
                n_samples_per_theta=n_calibration_points_per_theta,
                verbose=True,
                calibration_method='histogram',
                bins=50,
                interpolation='quadratic'
            )
            predictions[f'{model_name} Calibrated'] = dict(Contours=calibrated_contours, MLE=calibrated_mle)

    return predictions


def get_exact_contours(
        theta_0: np.ndarray,
        X_true: np.ndarray,
        predict_grid: ParamGrid
):
    p_0 = MultiDimToyModel(*theta_0)
    log_prob_0 = p_0.log_prob(X_true)

    @tf.function
    def nllr_exact(alpha, beta, X):
        p_theta = MultiDimToyModel(alpha=alpha, beta=beta)
        return -tf.keras.backend.sum((p_theta.log_prob(X) - log_prob_0))

    Alphas, Betas = predict_grid.meshgrid()
    exact_contours = np.zeros_like(Alphas)
    for i in range(Alphas.shape[0]):
        for j in range(Alphas.shape[1]):
            alpha = tf.constant(Alphas[i, j])
            beta = tf.constant(Betas[i, j])
            nllr = nllr_exact(alpha, beta, X_true)
            exact_contours[i, j] = nllr

    return exact_contours


def get_exact_mle(X_true):
    var_alpha = tf.Variable(tf.constant(1, dtype=tf.float32))
    var_beta = tf.Variable(tf.constant(-1, dtype=tf.float32))
    p_var = MultiDimToyModel(var_alpha, var_beta)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    n_iter = int(1e3)
    nll = tf.function(lambda: -tf.keras.backend.sum(p_var.log_prob(X_true)))

    for i in range(n_iter):
        optimizer.minimize(nll, [var_alpha, var_beta])

    return np.array([var_alpha.numpy(), var_beta.numpy()])


def plot_exclusion_contours(
        ax: Axes,
        contours: np.ndarray,
        mle: np.ndarray,
        exact_mle: np.ndarray,
        predict_grid: ParamGrid
):
    if not np.all(mle == exact_mle):
        ax.plot(mle[0], mle[1], 'bo', ms=8, label='Predicted')
    ax.plot(exact_mle[0], exact_mle[1], 'go', label='Exact', ms=8)
    ax.plot(ALPHA_TRUE, BETA_TRUE, 'ro', ms=8, label='True')

    test_stat = 2 * (contours - contours.min())
    im = ax.contourf(*predict_grid.meshgrid(), test_stat, levels=np.arange(0, 140, step=20))
    ax.contour(
        *predict_grid.meshgrid(),
        test_stat,
        colors=['w'],
        levels=[chi2.ppf(0.683, df=2),
                chi2.ppf(0.9545, df=2),
                chi2.ppf(0.9973, df=2)]
    )
    ax.legend()
    return im


def plot_predictions(
        predictions: Dict,
        predict_grid: ParamGrid
):
    fig, axarr = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    for ax, (model_name, pred) in zip(np.ravel(axarr), predictions.items()):
        im = plot_exclusion_contours(
            ax=ax,
            contours=pred['Contours'],
            mle=pred['MLE'],
            exact_mle=predictions['Exact']['MLE'],
            predict_grid=predict_grid,
        )
        ax.set_title(model_name)
        fig.colorbar(im, ax=ax)
    return fig


def run_single_experiment(
        n_samples_per_theta: int,
        theta_0: np.ndarray,
        n_theta_train: int,
        hyperparams: Dict,
        n_calibration_points_per_theta,
        n_theta_pred: int
):
    X_true = get_X_true()
    ds = create_dataset(theta_0=theta_0, n_samples_per_theta=n_samples_per_theta, n_theta_train=n_theta_train)
    models = create_models(**hyperparams)
    predict_grid = get_predict_grid(n_theta_pred=n_theta_pred)
    predictions = fit_predict_models(
        models=models,
        dataset=ds,
        X_true=X_true,
        predict_grid=predict_grid,
        n_calibration_points_per_theta=n_calibration_points_per_theta
    )
    exact_contours = get_exact_contours(theta_0=theta_0, X_true=X_true, predict_grid=predict_grid)
    exact_mle = get_exact_mle(X_true)
    predictions['Exact'] = dict(Contours=exact_contours, MLE=exact_mle)
    pred_plot = plot_predictions(predictions=predictions, predict_grid=predict_grid)
    return predictions, pred_plot


if __name__ == '__main__':
    set_all_random_seeds()
    matplotlib_setup()
    run_kwargs = dict(
        n_samples_per_theta=500,
        theta_0=np.array([0, 0]),
        n_theta_train=30,
        hyperparams=dict(
            epochs=20,
            patience=5,
            validation_split=0.1,
            n_hidden=(30, 30),
        ),
        n_calibration_points_per_theta=int(1e4),
        n_theta_pred=15,
    )
    predictions, pred_plot = run_single_experiment(**run_kwargs)
    save_results(
        'multidim',
        figures=dict(predictions=pred_plot),
        config=run_kwargs
    )