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


def _get_softmax_logits_from_binary_probs(probs: np.ndarray):
    probs = probs.squeeze()
    assert len(probs.shape) == 1
    logits = logit(probs)
    logits = np.stack([np.zeros_like(logits), logits]).T

    # TODO: move to a test file
    # reconstructed_probs = tf.keras.backend.softmax(logits, axis=1)[:, 1]
    # np.testing.assert_array_almost_equal(probs, reconstructed_probs)

    return logits


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

    score_names = ('Brier Score', 'F1 (Micro)', 'Expected Calibration Error')
    scores = pd.Series(index=pd.MultiIndex.from_product([score_names, ratio_models]))
    calibration_curves = []

    for model_name, model in ratio_models.items():
        y_prob = model.clf.predict_proba(dataset_sample.build_input())[:, 1]
        if y_prob.max() >= 1.0:
            y_prob = minmax_scale(y_prob)
        y_pred = np.around(y_prob)
        accuracy, confidence = calibration_curve(
            y_true=dataset_sample.y,
            y_prob=y_prob,
            n_bins=n_bins,
            strategy=strategy
        )
        calibration_curves.append(pd.Series(index=confidence, data=accuracy, name=model_name))

        scores['Brier Score', model_name] = brier_score_loss(
            y_true=dataset_sample.y,
            y_prob=y_prob
        )
        scores['F1 (Micro)', model_name] = f1_score(
            y_true=dataset_sample.y,
            y_pred=y_pred,
            average='micro'
        )
        scores['Expected Calibration Error', model_name] = tfp.stats.expected_calibration_error(
            num_bins=n_bins,
            logits=tf.cast(_get_softmax_logits_from_binary_probs(y_prob), tf.float32),
            labels_true=tf.cast(dataset_sample.y, tf.int32)
        ).numpy()

    calibration_curves = pd.concat(calibration_curves, axis=1)
    calibration_curves['Ideal'] = calibration_curves.index.values
    calibration_curves.index.name = 'Confidence'

    return calibration_curves, scores
