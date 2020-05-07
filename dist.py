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


if __name__ == '__main__':
    plot_dist()
