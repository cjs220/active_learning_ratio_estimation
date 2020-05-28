from abc import ABC
from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf

from active_learning_ratio_estimation.model.keras_models import RegularDense, FlipoutDense


class BaseWrapper(BaseEstimator, ClassifierMixin, ABC):

    def __init__(self,
                 n_hidden: Sequence[int] = (10, 10),
                 activation: str = 'relu',
                 loss: str = 'bce',
                 optimizer: str = 'adam',
                 run_eagerly: bool = False,
                 epochs: int = 10,
                 validation_split: float = 0.2,
                 patience: int = 2,
                 verbose: int = 2,
                 ):
        self.n_hidden = n_hidden
        self.activation = activation

        # compile arguments
        self.loss = loss
        self.optimizer = optimizer
        self.run_eagerly = run_eagerly

        # fit arguments
        self.epochs = epochs
        self.validation_split = validation_split
        self.patience = patience
        self.verbose = verbose

    def get_keras_model(self) -> tf.keras.Model:
        raise NotImplementedError

    def fit(self, X, y):
        self.n_samples_ = int((1-self.validation_split)*len(X))
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        assert self.n_classes_ == 2
        model = self.get_keras_model()
        metrics = ['accuracy']
        model.compile(loss=self.loss, optimizer=self.optimizer, run_eagerly=self.run_eagerly, metrics=metrics)
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True)]
        model.fit(X, y,
                  epochs=self.epochs,
                  callbacks=callbacks,
                  verbose=self.verbose,
                  validation_split=self.validation_split)
        self.model_ = model
        return self

    def predict_proba(self, X, **kwargs):
        probs = self.model_.predict(X, **kwargs)
        probs = np.hstack([1 - probs, probs])
        return probs

    def predict(self, X, **kwargs):
        return np.around(self.predict_proba(X, **kwargs)[:, 1])

    def score(self, X, y, sample_weight=None):
        # TODO
        pass


class BaseBayesianWrapper(BaseWrapper, ABC):

    def sample_predictive_distribution(self, X, samples=100, **kwargs):
        X_tile = np.repeat(X, samples, axis=0)
        probs = super().predict_proba(X_tile, **kwargs).reshape(len(X), samples, 2)
        return probs

    def predict_proba(self, X, samples=100, **kwargs):
        probs = self.sample_predictive_distribution(X, samples=samples, **kwargs)
        mean_probs = probs.mean(axis=1)
        return mean_probs


class DenseClassifier(BaseWrapper):

    def get_keras_model(self):
        return RegularDense(n_hidden=self.n_hidden, activation=self.activation)


class FlipoutClassifier(BaseBayesianWrapper):

    def get_keras_model(self):
        return FlipoutDense(n_hidden=self.n_hidden, activation=self.activation, n_samples=self.n_samples_)
