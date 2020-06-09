import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from active_learning_ratio_estimation.dataset import SinglyParameterizedRatioDataset, ParamGrid
from active_learning_ratio_estimation.model import FlipoutClassifier, SinglyParameterizedRatioModel, DenseClassifier
from experiments.mixtures import triple_mixture


def main():
    # params
    n_samples_per_theta = int(1e3)
    theta_0 = 0.0
    n_theta = 101
    hyperparams = dict(
        n_hidden=(15, 15),
        epochs=20,
        patience=5,
        validation_split=0.1,
        verbose=2,
    )
    theta_true = 0.05
    n_samples_true = 500

    # build dataset
    theta_1_grid = ParamGrid(bounds=[(0, 0.15)], num=n_theta)
    dataset = SinglyParameterizedRatioDataset.from_simulator(
        simulator_func=triple_mixture,
        theta_0=theta_0,
        theta_1_iterator=theta_1_grid,
        n_samples_per_theta=n_samples_per_theta,
        include_log_probs=True
    )

    # build model
    estimator = DenseClassifier(**hyperparams)
    ratio_model = SinglyParameterizedRatioModel(estimator=estimator)
    ratio_model.fit(dataset)

    # predict model
    p_true = triple_mixture(theta_true)
    X_true = p_true.sample(n_samples_true).numpy()
    nllr, mle = ratio_model.nllr_param_scan(
        x=X_true,
        param_grid=theta_1_grid,
        meshgrid_shape=False
    )
    nllr_cal, mle_cal = ratio_model.nllr_param_scan_with_calibration(
        x=X_true,
        param_grid=theta_1_grid,
        n_samples_per_theta=n_samples_per_theta,
        simulator_func=triple_mixture,
        calibration_method='sigmoid',
        meshgrid_shape=False,
        verbose=True
    )

    # calculate exact nllr
    nllr_exact = np.zeros_like(nllr)
    log_prob_0 = triple_mixture(theta_0).log_prob(X_true)
    for i, theta in enumerate(theta_1_grid):
        log_prob_1 = triple_mixture(theta.item()).log_prob(X_true)
        nllr_exact[i] = - (log_prob_1 - log_prob_0).numpy().sum()

    # plot results
    fig, ax = plt.subplots()
    results = pd.DataFrame(
        index=theta_1_grid.array.squeeze(),
        data= {'Exact': nllr_exact.squeeze(), 'Predicted': nllr.squeeze(), 'Predicted (Calibrated)': nllr_cal.squeeze()}
    )
    results.plot(ax=ax)
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    main()
