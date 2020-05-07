import os

import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf

tfd = tfp.distributions

from dataset import RatioDataset


class RatioModel:

    def __init__(self,
                 x_dim,
                 theta_dim,
                 hidden_dims=(10, 10),
                 activation='relu',
                 parameterised=True):
        self.parameterised = parameterised
        self._build_input = self._build_input_parameterised if parameterised else self._build_input_unparameterised
        input_dim = x_dim + 2 * theta_dim if parameterised else x_dim
        self.tf_model = self.default_tf_model(input_dim=input_dim,
                                              hidden_dims=hidden_dims,
                                              activation=activation)

    @staticmethod
    def default_tf_model(input_dim, hidden_dims=(10, 10), activation='relu'):
        tf_model = tf.keras.Sequential(layers=[
            tf.keras.layers.Dense(hidden_dims[0], input_dim=input_dim),
            tf.keras.layers.Activation(activation),
            tf.keras.layers.Dense(hidden_dims[1], input_dim=hidden_dims[0]),
            tf.keras.layers.Activation(activation),
            tf.keras.layers.Dense(1, input_dim=hidden_dims[1]),
            tf.keras.layers.Activation('sigmoid')
        ])
        return tf_model

    @property
    def default_callbacks(self):
        default_callbacks = [
            tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=10),
            tf.keras.callbacks.TensorBoard()
        ]
        return default_callbacks

    @property
    def default_loss(self):
        default_loss = tf.keras.losses.BinaryCrossentropy()
        return default_loss

    @property
    def default_optimizer(self):
        default_optimizer = tf.keras.optimizers.Adam()
        return default_optimizer

    def fit_dataset(self, ds: RatioDataset, epochs=10):
        df = ds.dataframe
        data = {quantity: df[quantity].values.astype(np.float32) for quantity in ('theta_0', 'theta_1', 'x', 'y')}
        self.fit(epochs=epochs, **data)

    def fit(self, theta_0, theta_1, x, y, epochs=10):
        self.tf_model.compile(optimizer=self.default_optimizer,
                              loss=self.default_loss,
                              metrics=['accuracy'])
        model_input = self._build_input(theta_0=theta_0, theta_1=theta_1, x=x)
        self.tf_model.fit(x=model_input,
                          y=y,
                          validation_split=0.2,
                          batch_size=32,
                          callbacks=self.default_callbacks,
                          epochs=epochs)

    def predict_proba(self, theta_0, theta_1, x):
        model_input = self._build_input(theta_0=theta_0, theta_1=theta_1, x=x)
        return self.tf_model.predict(model_input)

    @staticmethod
    def _build_input_parameterised(theta_0, theta_1, x):
        pass

    @staticmethod
    def _build_input_unparameterised(theta_0, theta_1, x):
        return np.atleast_2d(x).T



class BayesianRatioModel(RatioModel):

    # @property
    # def default_optimizer(self):

    # @property
    # def default_loss(self):
    #     return lambda y_true, model_out: tf.keras.losses.binary_crossentropy(y_true, model_out.probs)

    @staticmethod
    def default_tf_model(input_dim, hidden_dims=(10, 10), activation='relu'):
        num_samples = int(0.8 * 2e5)

        kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                                  tf.cast(num_samples, dtype=tf.float32))

        tf_model = tf.keras.Sequential(layers=[
            tfp.layers.DenseFlipout(hidden_dims[0],
                                    kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                                    bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                                    kernel_divergence_fn=kl_divergence_function,
                                    activation=activation),
            tfp.layers.DenseFlipout(hidden_dims[1],
                                    kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                                    bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                                    kernel_divergence_fn=kl_divergence_function,
                                    activation=activation),
            tfp.layers.DenseFlipout(1,
                                    kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                                    bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                                    kernel_divergence_fn=kl_divergence_function,
                                    activation='sigmoid'),
        ])
        return tf_model


def run_tensorboard():
    os.system('tensorboard --logdir=logs')


if __name__ == '__main__':
    from dist import triple_mixture, ideal_classifier_probs, negative_log_likelihood_ratio
    import matplotlib.pyplot as plt
    import pandas as pd

    simulator_func = triple_mixture
    theta_0 = 0.05
    theta_1 = 0

    ds = RatioDataset(
        n_thetas=1,
        n_samples_per_theta=int(1e5),
        simulator_func=simulator_func,
        theta_0_dist=theta_0,
        theta_1_dist=theta_1
    )

    f, axarr = plt.subplots(2)
    x = np.linspace(-5, 5, int(1e4))

    model_types = [
        ('Regular', RatioModel, 100),
        ('Bayesian', BayesianRatioModel, 100)
    ]

    df = pd.DataFrame(index=x)
    models = []

    for name, model_cls, epochs in model_types:
        clf = model_cls(x_dim=1,
                        theta_dim=1,
                        parameterised=False,
                        hidden_dims=(10, 10))

        clf.fit_dataset(ds, epochs=epochs)
        models.append(clf)

        y_pred = clf.tf_model.predict_proba(x)
        lr_estimate = y_pred / (1 - y_pred)
        nllr = -np.log(lr_estimate)

        df[f'y_pred ({name})'] = y_pred.squeeze()
        df[f'NLLR ({name})'] = nllr.squeeze()

    y_pred_ideal = ideal_classifier_probs(x, simulator_func, theta_0, theta_1)
    df['y_pred (Ideal)'] = y_pred_ideal
    df['NLLR (True)'] = negative_log_likelihood_ratio(x, simulator_func, theta_0, theta_1)

    for i, variable in enumerate(['y_pred', 'NLLR']):
        cols = list(filter(lambda x: variable in x, df.columns))
        df[cols].plot(ax=axarr[i])
    plt.show()
    # run_tensorboard()
