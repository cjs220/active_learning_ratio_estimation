from abc import ABC
from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf


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
        model = self.get_keras_model()
        metrics = ['accuracy']
        model.compile(loss=self.loss, optimizer=self.optimizer, run_eagerly=self.run_eagerly, metrics=metrics)
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True)]
        model.fit(X, y, callbacks=callbacks)
        self.model_ = model
        return self

    def predict_proba(self, X):
        # TODO
        raise AttributeError


class BaseBayesianClassifier(BaseWrapper, ABC):

    def sample_predict(self, X, samples=100):
        X_tile = np.repeat(X, samples, axis=0)

    def predict_proba(self, X, samples=100, return_std=False):
        x_tile = np.repeat(x, self.prediction_mc_samples, axis=0)
        preds = super(BaseBayesianFeedForward, self).predict_proba(x_tile, **kwargs).squeeze()
        stack_preds = np.stack(np.split(preds, len(x)))
        y_pred = stack_preds.mean(axis=1)
        return y_pred.reshape(-1, 1)




class DenseClassifier(BaseWrapper):

    def get_keras_model(self):
        return