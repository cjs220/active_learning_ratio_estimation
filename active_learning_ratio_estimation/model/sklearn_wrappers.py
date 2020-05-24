from abc import ABC
from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf

from active_learning_ratio_estimation.model.keras_models import FeedForward, FlipoutFeedForward


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
                 ):
        self.n_hidden = n_hidden
        self.activation = activation

        self.loss = loss
        self.optimizer = optimizer
        self.run_eagerly = run_eagerly

        self.epochs = epochs
        self.validation_split = validation_split
        self.patience = patience

    def get_keras_model(self):
        raise NotImplementedError

    def fit(self, X, y):
        self.n_samples_ = int((1-self.validation_split)*len(X))
        model = self.get_keras_model()
        metrics = ['accuracy']
        model.compile(loss=self.loss, optimizer=self.optimizer, run_eagerly=self.run_eagerly, metrics=metrics)
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True)]
        model.fit(X, y, callbacks=callbacks)
        self.model_ = model
        return self

    def predict_proba(self, X):
        probs = self.model_.predict(X)
        probs = np.hstack([1 - probs, probs])
        return probs

    def score(self, X, y, sample_weight=None):
        # TODO
        pass


class BaseBayesianClassifier(BaseWrapper, ABC):

    def sample_predictive_distribution(self, X, samples=100):
        X_tile = np.repeat(X, samples, axis=0)
        probs = super().predict_proba(X_tile)
        probs = np.stack(np.split(probs, len(X)))
        return probs

    def predict_proba(self, X, samples=100, return_std=False):
        probs = self.sample_predictive_distribution(X, samples=samples)
        mean_probs = probs.mean(axis=1)
        return mean_probs


class DenseClassifier(BaseWrapper):

    def get_keras_model(self):
        return FeedForward(n_hidden=self.n_hidden, activation=self.activation)


class FlipoutClassifier(BaseBayesianClassifier):

    def get_keras_model(self):
        return FlipoutFeedForward(n_hidden=self.n_hidden, activation=self.activation, n_samples=self.n_samples_)
