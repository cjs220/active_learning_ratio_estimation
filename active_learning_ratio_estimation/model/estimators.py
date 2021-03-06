from abc import ABC
from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from active_learning_ratio_estimation.model.keras_models import RegularDense, FlipoutDense


class BaseWrapper(BaseEstimator, ABC):

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
                 predict_batch_size: int = 32,
                 label_smoothing: float = 0.0,
                 ):
        self.n_hidden = n_hidden
        self.scale_input = scale_input
        self.activation = activation

        # compile arguments
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                       label_smoothing=label_smoothing)
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

    def fit(self, X: np.ndarray, y: np.ndarray):
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
        self.history_ = model.fit(
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

    def predict_logits(self, X: np.ndarray, **predict_params):
        if self.scale_input:
            X = self.scaler_.transform(X)

        batch_size = predict_params.get('batch_size', None) or self.predict_batch_size
        if batch_size == -1:
            batch_size = len(X)

        logits = self.model_.predict(X, batch_size, **predict_params)
        return logits

    def predict_proba(self, X: np.ndarray, **predict_params):
        logits = self.predict_logits(X=X, **predict_params)
        probs = tf.nn.sigmoid(logits).numpy()
        probs = np.hstack([1 - probs, probs])
        return probs

    def predict(self, X: np.ndarray, **predict_params):
        return np.around(self.predict_proba(X, **predict_params)[:, 1])


class BaseBayesianWrapper(BaseWrapper, ABC):

    def sample_predictive_distribution(self, X: np.ndarray, samples: int = 100, **predict_params) -> np.ndarray:
        X_tile = np.repeat(X, samples, axis=0)
        logits_samples = super().predict_logits(X_tile, **predict_params).reshape(len(X), samples)
        return logits_samples

    def predict_logits(self, X, samples=100, return_std=False, **predict_params) -> np.ndarray:
        logits_samples = self.sample_predictive_distribution(X, samples=samples, **predict_params)
        mean_logits = np.median(logits_samples, axis=1)
        if return_std:
            std = logits_samples.std(axis=1, ddof=1)
            return mean_logits, std
        else:
            return mean_logits

    def predict_proba(self, X: np.ndarray, samples=100, return_std=False, **predict_params):
        logits_samples = self.sample_predictive_distribution(X=X, samples=samples, **predict_params)
        probs_samples = tf.nn.sigmoid(logits_samples).numpy()
        probs = np.median(probs_samples, axis=1)
        probs = np.stack([1 - probs, probs], axis=1)
        if return_std:
            std = probs_samples.std(axis=1, ddof=1)
            return probs, std
        else:
            return probs

    def predict(self, X: np.ndarray, **predict_params):
        raise NotImplementedError


class DenseClassifier(BaseWrapper):

    def __init__(self,
                 l2_regularization: float = 0.0,
                 n_hidden: Sequence[int] = (10, 10),
                 scale_input: bool = True,
                 activation: str = 'tanh',
                 optimizer: str = 'adam',
                 run_eagerly: bool = False,
                 epochs: int = 10,
                 fit_batch_size: int = 32,
                 validation_split: float = 0.2,
                 validation_batch_size: int = 32,
                 patience: int = 2,
                 verbose: int = 2,
                 predict_batch_size: int = 32,
                 label_smoothing: float = 0.0,
                 ):
        self.l2_regularization = l2_regularization
        super().__init__(
            n_hidden=n_hidden,
            scale_input=scale_input,
            activation=activation,
            optimizer=optimizer,
            run_eagerly=run_eagerly,
            epochs=epochs,
            fit_batch_size=fit_batch_size,
            validation_split=validation_split,
            validation_batch_size=validation_batch_size,
            patience=patience,
            verbose=verbose,
            predict_batch_size=predict_batch_size,
            label_smoothing=label_smoothing
        )

    def get_keras_model(self):
        return RegularDense(
            n_hidden=self.n_hidden,
            activation=self.activation,
            l2_regularization=self.l2_regularization
        )


class FlipoutClassifier(BaseBayesianWrapper):

    def get_keras_model(self):
        return FlipoutDense(n_hidden=self.n_hidden, activation=self.activation, n_samples=self.n_samples_)
