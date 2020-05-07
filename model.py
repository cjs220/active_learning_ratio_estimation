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
            tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=2),
            tf.keras.callbacks.TensorBoard()
        ]
        return default_callbacks

    @property
    def default_loss(self):
        default_loss = tf.keras.losses.BinaryCrossentropy()
        return default_loss

    @property
    def default_optimizer(self):
        # default_optimizer = optimizers.Adam()
        default_optimizer = 'rmsprop'
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


def prior_trainable(kernel_size, bias_size=0, dtype=None):
    # Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(
            lambda t: tfd.Independent(
                tfd.Normal(loc=t, scale=1), reinterpreted_batch_ndims=1)),
    ])


def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    # Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(
            lambda t: tfd.Independent(
                tfd.Normal(loc=t[..., :n], scale=1e-5 + tf.nn.softplus(c + t[..., n:])), reinterpreted_batch_ndims=1)),
    ])


class BayesianRatioModel(RatioModel):

    @property
    def default_loss(self):
        return lambda y_true, model_out: tf.keras.losses.binary_crossentropy(y_true, model_out.probs)

    @staticmethod
    def default_tf_model(input_dim, hidden_dims=(10, 10), activation='relu'):
        tf_model = tf.keras.Sequential(layers=[
            tfp.layers.DenseVariational(hidden_dims[0],
                                        make_posterior_fn=posterior_mean_field,
                                        make_prior_fn=prior_trainable),
            tf.keras.layers.Activation(activation),
            tfp.layers.DenseVariational(hidden_dims[1],
                                        make_posterior_fn=posterior_mean_field,
                                        make_prior_fn=prior_trainable),
            tf.keras.layers.Activation(activation),
            tfp.layers.DenseVariational(1,
                                        make_posterior_fn=posterior_mean_field,
                                        make_prior_fn=prior_trainable),
            tfp.layers.DistributionLambda(lambda t: tfd.Bernoulli(logits=t))
        ])
        return tf_model


def run_tensorboard():
    os.system('tensorboard --logdir=logs')


if __name__ == '__main__':
    from dist import triple_mixture
    import matplotlib.pyplot as plt

    ds = RatioDataset(
        n_thetas=1,
        n_samples_per_theta=int(1e5),
        simulator_func=triple_mixture,
        theta_0_dist=0.05,
        theta_1_dist=0
    )

    f, axarr = plt.subplots(2)
    x = np.linspace(-5, 5, int(1e4))

    model_types = [
        ('Regular', RatioModel),
        ('Bayesian', BayesianRatioModel)
    ]

    for name, model_cls in model_types:
        clf = model_cls(x_dim=1, theta_dim=1, parameterised=False, hidden_dims=(10, 10))
        clf.fit_dataset(ds, epochs=100)
        clf.tf_model.predict_classes(x)
        y_pred_proba = clf.tf_model.predict_proba(x)
        y_pred = clf.tf_model.predict_classes(x)
        lr_estimate = y_pred_proba / (1 - y_pred_proba)
        nll = -np.log(lr_estimate)
        axarr[0].plot(x, y_pred_proba, label=f'Probability ({name})')
        axarr[0].plot(x, y_pred, label=f'Class ({name})')
        axarr[1].plot(x, nll, label=f'NLLR ({name})')
    axarr[0].legend()
    axarr[1].legend()
    plt.show()
    # run_tensorboard()
