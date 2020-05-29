import logging
from numbers import Number
from typing import Union, Callable

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
import tensorflow as tf

from active_learning_ratio_estimation.active_learning.acquisition_functions import acquisition_functions
from active_learning_ratio_estimation.dataset import ParamGrid, SinglyParameterizedRatioDataset, SingleParamIterator, \
    ParamIterator, build_singly_parameterized_input
from active_learning_ratio_estimation.model import SinglyParameterizedRatioModel
from active_learning_ratio_estimation.util import ensure_array, ideal_classifier_probs


logger = logging.getLogger(__name__)


def _get_best_epoch_information(keras_model: tf.keras.Model):
    history = pd.DataFrame(keras_model.history.history)
    best_epoch = history[history['val_loss'] == history['val_loss'].max()]
    return best_epoch.squeeze().to_dict()


class ActiveLearner:
    def __init__(self,
                 simulator_func: Callable,
                 theta_0: Union[Number, np.ndarray],
                 theta_1_iterator: ParamIterator,
                 n_samples_per_theta: int,
                 ratio_model: SinglyParameterizedRatioModel,
                 total_param_grid: ParamGrid,
                 test_dataset: SinglyParameterizedRatioDataset = None,
                 acquisition_function: Union[str, Callable] = 'entropy',
                 ucb_kappa: float = 1.0,
                 ):
        self.theta_0 = theta_0
        self.n_samples_per_theta = n_samples_per_theta
        self.dataset = SinglyParameterizedRatioDataset.from_simulator(
            simulator_func=simulator_func,
            theta_0=theta_0,
            theta_1_iterator=theta_1_iterator,
            n_samples_per_theta=n_samples_per_theta,
        )
        self.ratio_model = ratio_model
        self.model_fit()
        self.param_grid = total_param_grid
        self.trialed_thetas = [ensure_array(theta) for theta in np.unique(self.dataset.theta_1s)]
        self.simulator_func = simulator_func
        self.test_dataset = test_dataset
        if test_dataset is not None:
            if test_dataset.log_prob_0 is None or test_dataset.log_prob_1 is None:
                raise RuntimeError('Test dataset must have log probabilities of data points; '
                                   'pass include_log_probs=True to its from_simulator constructor')
        self._train_history = []
        self._test_history = []
        if isinstance(acquisition_function, str) and acquisition_function != 'random':
            acquisition_function = acquisition_functions[acquisition_function]
        self.acquisition_function = acquisition_function
        self.ucb_kappa = ucb_kappa
        self.gp = None
        self.gp_history = []

    def fit(self, n_iter: int):
        for i in range(n_iter):
            logger.info(f'Active learning iteration {i +1}/{n_iter}')
            self.step()
        return self

    def model_fit(self):
        self.ratio_model.fit(self.dataset)

    def model_eval(self):
        probs = self.ratio_model.predict_proba_dataset(self.test_dataset)
        l0, l1 = map(np.exp, [self.test_dataset.log_prob_0, self.test_dataset.log_prob_1])
        ideal_probs = ideal_classifier_probs(l0, l1)
        ideal_probs = np.hstack([1 - ideal_probs, ideal_probs])
        squared_error = (probs - ideal_probs) ** 2
        return squared_error.mean()

    @property
    def remaining_thetas(self):
        remaining_thetas = [theta for theta in self.param_grid.values if theta not in self.trialed_thetas]
        return remaining_thetas

    @property
    def test_history(self):
        return pd.DataFrame(self._test_history)

    @property
    def train_history(self):
        return pd.DataFrame(self._train_history)

    def step(self):
        next_theta = self.choose_theta()
        logger.info(f'Adding theta = {next_theta} to labeled data.')
        new_ds = SinglyParameterizedRatioDataset.from_simulator(
            simulator_func=self.simulator_func,
            theta_0=self.theta_0,
            theta_1_iterator=SingleParamIterator(next_theta),
            n_samples_per_theta=self.n_samples_per_theta,
        )
        self.dataset += new_ds
        self.dataset.shuffle()

        logger.info('Fitting ratio model')
        self.model_fit()
        training_info = _get_best_epoch_information(self.ratio_model.keras_model_)
        self._train_history.append(training_info)
        logger.info('Finished fitting ratio model. Best epoch information: '
                    + ', '.join([f'{name}={val:.2E}' for name, val in training_info.items()]))

        if self.test_dataset is not None:
            logger.info('Evaluating MSE on test dataset')
            mse = self.model_eval()
            logger.info(f'Test MSE: {mse:.2E}')
            self._test_history.append(dict(mse=mse))
        self.trialed_thetas.append(next_theta)

    def calculate_marginalised_acquisition(self):
        U_theta = []
        for theta in self.trialed_thetas:
            mask = self.dataset.theta_1s == theta
            x = self.dataset.x[mask]
            # TODO: the following section assumes that the ratio model that
            #  a) is Bayesian
            #  b) has as standard scaler
            #  But this is not ideal as we might want to test some of the acquisition functions with a regular NN
            theta_1s = self.dataset.theta_1s[mask]
            model_input = build_singly_parameterized_input(x=x, theta_1s=theta_1s)
            scaler, clf = self.ratio_model.estimator
            model_input = scaler.transform(model_input)
            sampled_probs = clf.sample_predictive_distribution(model_input)

            U_theta_x = self.acquisition_function(sampled_probs)
            assert U_theta_x.shape == (len(x),)
            U_theta.append(U_theta_x.mean())
        return U_theta

    def choose_theta(self):
        if self.acquisition_function == 'random':
            next_theta = self.remaining_thetas[np.random.randint(0, len(self.remaining_thetas))]
        else:
            length_scale = np.array([linspace[-1] - linspace[0] for linspace in self.param_grid.linspaces])
            rbf = RBF(length_scale=length_scale/10, length_scale_bounds='fixed')
            # rbf = RBF()
            white_kernel = WhiteKernel()
            self.gp = GaussianProcessRegressor(kernel=1*rbf + white_kernel,
                                               alpha=0,
                                               n_restarts_optimizer=5,
                                               normalize_y=True)

            X_train = np.array(self.trialed_thetas)
            y_train = np.array(self.calculate_marginalised_acquisition())
            self.gp.fit(X_train, y_train)

            X_test = np.array(self.remaining_thetas)
            mean, std = self.gp.predict(X_test, return_std=True)
            ucb = mean + self.ucb_kappa * std

            self.gp_history.append(dict(
                gp=clone(self.gp),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                mean=mean,
                std=std
            ))
            next_theta = self.remaining_thetas[np.argmax(ucb)]

        return next_theta


def _gp_debug(X_train, X_test, y_train, mean, std):
    plt.figure()
    plt.plot(X_train, y_train, 'ro')
    plt.plot(X_test, mean)
    plt.plot(X_test, mean - std, 'g')
    plt.plot(X_test, mean + std, 'g')
    plt.show()
