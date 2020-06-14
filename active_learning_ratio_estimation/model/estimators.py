from abc import ABC
from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from active_learning_ratio_estimation.model.keras_models import RegularDense, FlipoutDense


class BaseWrapper(BaseEstimator, ClassifierMixin, ABC):

    def __init__(self,
                 n_hidden: Sequence[int] = (10, 10),
                 scale_input: bool = True,
                 activation: str = 'relu',
                 optimizer: str = 'adam',
                 run_eagerly: bool = False,
                 epochs: int = 10,
                 fit_batch_size: int = 32,
                 validation_split: float = 0.2,
                 validation_batch_size: int = 32,
                 patience: int = 2,
                 verbose: int = 2,
                 predict_batch_size: int = -1,
                 ):
        self.n_hidden = n_hidden
        self.scale_input = scale_input
        self.activation = activation

        # compile arguments
        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.optimizer = optimizer
        self.run_eagerly = run_eagerly

        # fit arguments
        self.epochs = epochs
        self.fit_batch_size = fit_batch_size
        self.validation_split = validation_split
        self.validation_batch_size = validation_batch_size

        self.patience = patience
        self.verbose = verbose

        # predict arguments
        self.predict_batch_size = predict_batch_size

    def get_keras_model(self) -> tf.keras.Model:
        raise NotImplementedError

    def fit(self, X, y):
        if self.scale_input:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)

        self.n_samples_ = int((1 - self.validation_split) * len(X))
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        assert self.n_classes_ == 2

        model = self.get_keras_model()
        metrics = ['accuracy']
        model.compile(loss=self.loss, optimizer=self.optimizer, run_eagerly=self.run_eagerly, metrics=metrics)
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True)]
        model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.fit_batch_size,
            callbacks=callbacks,
            verbose=self.verbose,
            validation_split=self.validation_split,
            validation_batch_size=self.validation_batch_size
        )
        self.model_ = model
        return self

    def predict_proba(self, X, **predict_params):
        if self.scale_input:
            X = self.scaler_.transform(X)

        batch_size = predict_params.get('batch_size', None) or self.predict_batch_size
        if batch_size == -1:
            batch_size = len(X)

        probs = self.model_.predict(X, batch_size, **predict_params)
        probs = np.hstack([1 - probs, probs])
        return probs

    def predict(self, X, batch_size=-1, **kwargs):
        return np.around(self.predict_proba(X, batch_size=-batch_size, **kwargs)[:, 1])

    def score(self, X, y, sample_weight=None):
        # TODO
        pass


class BaseBayesianWrapper(BaseWrapper, ABC):

    def sample_predictive_distribution(self, X, samples=100, **predict_params):
        X_tile = np.repeat(X, samples, axis=0)
        probs = super().predict_proba(X_tile, **predict_params).reshape(len(X), samples, 2)
        return probs

    def predict_proba(self, X, samples=100, return_std=False, **predict_params):
        probs = self.sample_predictive_distribution(X, samples=samples, **predict_params)
        mean_probs = probs.mean(axis=1)
        if return_std:
            std = probs.std(axis=1, ddof=1)
            return mean_probs, std
        else:
            return mean_probs


class DenseClassifier(BaseWrapper):

    def get_keras_model(self):
        return RegularDense(n_hidden=self.n_hidden, activation=self.activation)


class FlipoutClassifier(BaseBayesianWrapper):

    def get_keras_model(self):
        return FlipoutDense(n_hidden=self.n_hidden, activation=self.activation, n_samples=self.n_samples_)
