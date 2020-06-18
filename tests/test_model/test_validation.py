import pytest
import numpy as np
import tensorflow as tf

from active_learning_ratio_estimation.model.validation import softmax_logits_from_binary_probs


def _generate_y_prob(size, n):
    return [np.random.rand(size) for _ in range(n)]


@pytest.mark.parametrize('probs', _generate_y_prob(10, 10))
def test_softmax_logits_from_binary_probs(probs):
    logits = softmax_logits_from_binary_probs(probs)
    reconstructed_probs = tf.keras.backend.softmax(logits, axis=1)[:, 1]
    np.testing.assert_array_almost_equal(probs, reconstructed_probs)
