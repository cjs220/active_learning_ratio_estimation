from typing import Dict, Union

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import logit
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, f1_score

from active_learning_ratio_estimation.dataset import RatioDataset
from active_learning_ratio_estimation.model import RatioModel


def _get_softmax_logits_from_binary_probs(probs: np.ndarray):
    probs = probs.squeeze()
    assert len(probs.shape) == 1
    logits = logit(probs)
    logits = np.stack([np.zeros_like(logits), logits]).T

    # TODO: move to a test file
    reconstructed_probs = tf.keras.backend.softmax(logits, axis=1)[:, 1]
    np.testing.assert_array_almost_equal(probs, reconstructed_probs)

    return logits


def plot_calibration(ratio_models: Union[Dict[str, RatioModel], RatioModel],
                     dataset: RatioDataset,
                     n_data: int = -1,
                     n_bins: int = 10,
                     strategy: str = 'uniform',
                     ):

    if isinstance(ratio_models, RatioModel):
        ratio_models = dict(Model=ratio_models)

    n_data = n_data if n_data != -1 else len(dataset)
    dataset.shuffle()
    dataset_sample = dataset[:n_data]

    probs = dict()
    preds = dict()
    logits = dict()
    for model_name, model in ratio_models.items():
        y_prob = model.predict_proba_dataset(dataset_sample)[:, 1]
        probs[model_name] = y_prob
        preds[model_name] = np.around(probs[model_name])
        logits[model_name] = _get_softmax_logits_from_binary_probs(y_prob)

    calibration_curves = dict()
    xmin, xmax = (0, 1)  # plot boundaries
    for model_name, y_prob in probs.items():
        accuracy, confidence = calibration_curve(
            y_true=dataset_sample.y,
            y_prob=y_prob,
            n_bins=n_bins,
            strategy=strategy
        )
        xmax = min(accuracy.max(), xmax)
        xmin = max(accuracy.min(), xmin)
        calibration_curves[model_name] = accuracy, confidence
        plt.plot(confidence, accuracy, 's-', label=model_name)
    plt.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), 'k--', label='Perfectly Calibrated')
    plt.ylim(min(map(min, probs.values())), max(map(max, probs.values())))
    plt.xlim(xmin - 0.1, xmax + 0.1)
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Curve')
    plt.legend()
    plt.tight_layout()

    brier_scores = dict()
    f1_scores = dict()
    eces = dict()
    for model_name in probs.keys():
        brier_scores[model_name] = brier_score_loss(y_true=dataset_sample.y, y_prob=probs[model_name])
        f1_scores[model_name] = f1_score(y_true=dataset_sample.y, y_pred=preds[model_name], average='binary')
        eces[model_name] = tfp.stats.expected_calibration_error(
            num_bins=n_bins,
            logits=tf.cast(logits[model_name], tf.float32),
            labels_true=tf.cast(dataset_sample.y, tf.int32)
        ).numpy()

    scores = {
        'Brier Score': brier_scores,
        'F1 Score': f1_scores,
        'Expected Calibration Error': eces
    }

    return calibration_curves, scores
