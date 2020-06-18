from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, f1_score
from scipy.special import logit
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.preprocessing import minmax_scale

from active_learning_ratio_estimation.dataset import RatioDataset
from active_learning_ratio_estimation.model import BaseRatioModel


def get_calibration_metrics(
        ratio_models: Union[Dict[str, BaseRatioModel], BaseRatioModel],
        dataset: RatioDataset,
        n_data: int = -1,
        n_bins: int = 10,
        strategy: str = 'uniform',
):
    if isinstance(ratio_models, BaseRatioModel):
        ratio_models = dict(Model=ratio_models)

    # shuffle and slice data
    n_data = n_data if n_data != -1 else len(dataset)
    dataset.shuffle()
    dataset_sample = dataset[:n_data]
    y_true = dataset_sample.y

    score_names = ('Brier Score', 'F1 (Micro)', 'Expected Calibration Error')
    scores = pd.Series(index=pd.MultiIndex.from_product([score_names, ratio_models]))
    calibration_curves = []

    for model_name, model in ratio_models.items():
        y_prob = model.clf.predict_proba(dataset_sample.build_input())[:, 1]
        y_prob = _normalize_y_prob(y_prob)
        y_pred = np.around(y_prob)
        accuracy, confidence = calibration_curve(
            y_true=y_true,
            y_prob=y_prob,
            n_bins=n_bins,
            strategy=strategy
        )
        calibration_curves.append(pd.Series(index=confidence, data=accuracy, name=model_name))

        scores['Brier Score', model_name] = calculate_brier_score(y_prob=y_prob, y_true=y_true)
        scores['F1 (Micro)', model_name] = calculate_f1_micro(y_pred=y_pred, y_true=y_true)
        scores['Expected Calibration Error', model_name] = \
            calculate_expected_calibration_error(y_prob=y_prob, y_true=y_true, n_bins=n_bins)

    calibration_curves = pd.concat(calibration_curves, axis=1)
    calibration_curves['Ideal'] = calibration_curves.index.values
    calibration_curves.index.name = 'Confidence'

    return calibration_curves, scores


def calculate_brier_score(y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    _assert_one_d(y_prob)
    return brier_score_loss(y_true=y_true, y_prob=y_prob)


def calculate_f1_micro(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    _assert_one_d(y_pred)
    return f1_score(y_true=y_true, y_pred=y_pred, average='micro')


def calculate_expected_calibration_error(y_prob: np.ndarray, y_true: np.ndarray, n_bins: int):
    _assert_one_d(y_prob)
    ece = tfp.stats.expected_calibration_error(
        num_bins=n_bins,
        logits=tf.cast(softmax_logits_from_binary_probs(y_prob), tf.float32),
        labels_true=tf.cast(y_true, tf.int32)
    ).numpy()
    return ece


def softmax_logits_from_binary_probs(probs: np.ndarray) -> np.ndarray:
    # gets the equivalent softmax logits from the probability output of binary classifier
    _assert_one_d(probs)
    probs = probs.squeeze()
    logits = logit(probs)
    logits = np.stack([np.zeros_like(logits), logits]).T
    return logits


def _normalize_y_prob(y_prob: np.ndarray) -> np.ndarray:
    # make sure y_prob is in range [0, 1]
    if y_prob.max() >= 1.0 or y_prob.min() <= 0.0:
        return minmax_scale(y_prob)
    else:
        return y_prob


def _assert_one_d(arr: np.ndarray):
    assert arr.squeeze().ndim == 1
