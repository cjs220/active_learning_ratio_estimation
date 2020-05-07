import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

tfd = tfp.distributions


def triple_mixture(gamma):
    mixture_probs = [
        0.5 * (1 - gamma),
        0.5 * (1 - gamma),
        gamma
    ]
    gaussians = [
        tfd.Normal(loc=-2, scale=0.75),
        tfd.Normal(loc=0, scale=2),
        tfd.Normal(loc=1, scale=0.5)
    ]
    dist = tfd.Mixture(
        cat=tfd.Categorical(probs=mixture_probs),
        components=gaussians
    )
    return dist


def plot_dist():
    for gamma in (0, 0.05):
        dist = triple_mixture(gamma=gamma)
        x = np.linspace(-5, 5, int(1e4))
        p_x = dist.prob(x).numpy()
        plt.plot(x, p_x, label=fr'$\gamma = {gamma}$')
    plt.tight_layout()
    plt.legend()
    plt.show()


def _get_likelihoods(x, simulator_func, theta_0, theta_1):
    dist_0 = simulator_func(theta_0)
    dist_1 = simulator_func(theta_1)
    l0 = dist_0.prob(x)
    l1 = dist_1.prob(x)
    return l0, l1


def ideal_classifier_probs(x, simulator_func, theta_0, theta_1):
    l0, l1 = _get_likelihoods(x=x, simulator_func=simulator_func, theta_0=theta_0, theta_1=theta_1)
    return l1/(l0+l1)


def likelihood_ratio(x, simulator_func, theta_0, theta_1):
    l0, l1 = _get_likelihoods(x=x, simulator_func=simulator_func, theta_0=theta_0, theta_1=theta_1)
    return l1/l0


def negative_log_likelihood_ratio(x, simulator_func, theta_0, theta_1):
    return -np.log(likelihood_ratio(x=x, simulator_func=simulator_func, theta_0=theta_0, theta_1=theta_1))


if __name__ == '__main__':
    plot_dist()
