import numpy as np
import matplotlib.pyplot as plt


def bayesian_optimization_surrogate():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 400)
    truth = np.sin(x) + 0.1 * x
    sample_x = np.array([1.0, 4.0, 7.0, 9.0])
    sample_y = np.sin(sample_x) + 0.1 * sample_x
    weights = np.exp(-0.5 * ((x[:, None] - sample_x[None, :]) / 1.2) ** 2)
    mean = weights @ sample_y / weights.sum(axis=1)
    std = 0.6 * (1 - weights.max(axis=1))
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    top.plot(x, truth, "k--", alpha=0.5, label="unknown objective")
    top.plot(x, mean, "b", label="surrogate mean")
    top.fill_between(x, mean - std, mean + std, alpha=0.2, label="uncertainty")
    top.scatter(sample_x, sample_y, color="red", zorder=3, label="observations")
    top.legend(fontsize=8); top.set_ylabel("$f(x)$"); top.set_title("Bayesian optimization: surrogate and acquisition")
    acquisition = std - (mean - mean.min())
    bottom.plot(x, acquisition, "g")
    bottom.scatter([x[int(np.argmax(acquisition))]], [acquisition.max()], color="green", marker="*", s=180)
    bottom.set_xlabel("$x$"); bottom.set_ylabel("acquisition"); bottom.grid(True, alpha=0.3)
    plt.tight_layout()
    return "030803_bayesian_optimization_surrogate"


bayesian_optimization_surrogate()
plt.savefig('030803_bayesian_optimization_surrogate.png', dpi=110)
