import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split

from active_learning_ratio_estimation.model.estimators import BaseWrapper


class TemperatureCalibratedClassifier:

    def __init__(self, clf: BaseWrapper, validation_split: float, shuffle: bool = True):
        self.clf = clf
        self.validation_split = validation_split
        self.shuffle = shuffle

    def fit(self, X, y):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                              test_size=self.validation_split,
                                                              shuffle=self.shuffle)
        self.clf.fit(X_train, y_train)
        logits = tf.constant(self.clf.model_(X_valid).numpy().squeeze())
        labels = tf.constant(y_valid, dtype=tf.float32)

        @tf.function
        def _loss(T):
            scaled_logits = tf.multiply(logits, 1.0 / T)
            return tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=scaled_logits, labels=labels),
                axis=0
            )

        optim_results = tfp.optimizer.lbfgs_minimize(
            lambda T: tfp.math.value_and_gradient(_loss, T),
            initial_position=tf.ones(shape=(1,))
        )
        self.T_ = optim_results.position.numpy().item()
        return self

    def predict_logits(self, X, **predict_kwargs):
        logits = self.clf.predict_logits(X, **predict_kwargs)
        scaled_logits = logits * 1 / self.T_
        return scaled_logits

    def predict_proba(self, X, **predict_kwargs):
        scaled_logits = self.predict_logits(X, **predict_kwargs)
        probs = tf.nn.sigmoid(scaled_logits).numpy()
        probs = np.hstack([1 - probs, probs])
        return probs


class ParameterizedTemperatureCalibrator:

    def fit(self, logits, theta, y):
        raise NotImplementedError

    def predict(self, logits, theta):
        raise NotImplementedError


if __name__ == '__main__':
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import ShuffleSplit
    from sklearn.datasets import make_blobs
    from active_learning_ratio_estimation.model import (
        calculate_expected_calibration_error, DenseClassifier, calculate_brier_score
    )

    n_samples = 50000
    n_bins = 3  # use 3 bins for calibration_curve as we have 3 clusters here

    # Generate 3 blobs with 2 classes where the second blob contains
    # half positive samples and half negative samples. Probability in this
    # blob is therefore 0.5.
    centers = [(-5, -5), (0, 0), (5, 5)]
    X, y = make_blobs(n_samples=n_samples, centers=centers, shuffle=False,
                      random_state=42)
    y[:n_samples // 2] = 0
    y[n_samples // 2:] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    hyperparams = dict(
        n_hidden=(10, 10),
        epochs=1,
        verbose=0,
        run_eagerly=True,
    )
    regular_clf = DenseClassifier(**hyperparams)
    temp_clf = TemperatureCalibratedClassifier(DenseClassifier(**hyperparams),
                                               validation_split=0.25)
    sigmoid_clf = CalibratedClassifierCV(DenseClassifier(**hyperparams),
                                          method='sigmoid',
                                          cv=ShuffleSplit(n_splits=2, test_size=0.25))

    for clf_name, clf in [('Regular', regular_clf), ('Temp', temp_clf), ('Sigmoid', sigmoid_clf)]:
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        ece = calculate_expected_calibration_error(y_prob=y_prob, y_true=y_test, n_bins=10)
        brier = calculate_brier_score(y_prob=y_prob, y_true=y_test)
        print(f'{clf_name}: Brier={brier}, ECE={ece}')

    pass
