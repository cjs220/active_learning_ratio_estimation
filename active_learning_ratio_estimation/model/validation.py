from typing import Dict, Union

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from active_learning_ratio_estimation.dataset import RatioDataset
from active_learning_ratio_estimation.model import RatioModel


def plot_calibration(ratio_models: Union[Dict[str, RatioModel], RatioModel],
                     dataset: RatioDataset,
                     n_data: int = -1,
                     n_bins: int = 10,
                     strategy: str = 'uniform',
                     ):
    f, axarr = plt.subplots(2)

    if isinstance(ratio_models, RatioModel):
        ratio_models = dict(Model=ratio_models)

    n_data = n_data if n_data != -1 else len(dataset)
    dataset_sample = dataset.shuffle()[:n_data]

    probs = dict()
    for model_name, model in ratio_models.items():
        probs[model_name] = model.predict_proba_dataset(dataset_sample)[:, 1]

    calibration_curves = dict()
    for model_name, y_prob in probs.items():
        prob_true, prob_pred = calibration_curve(
            y_true=dataset_sample.y,
            y_prob=y_prob,
            n_bins=n_bins,
            strategy=strategy
        )
        calibration_curves[model_name] = prob_true, prob_pred
        axarr[0].plot(prob_true, prob_pred, 's-', label=model_name)
    axarr[0].plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), 'k--', label='Perfectly Calibrated')
    axarr[0].set_xlim(min(map(min, probs.values())), max(map(max, probs.values())))
    axarr[0].set_xlabel('Prob True')
    axarr[0].set_ylabel('Prob Pred')
    axarr[0].set_title('Reliability Curve')
    axarr[0].legend()
