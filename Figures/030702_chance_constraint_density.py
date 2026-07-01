import numpy as np
import matplotlib.pyplot as plt


def chance_constraint_density():
    mean, sigma = 100.0, 15.0
    threshold = mean + 1.645 * sigma
    x = np.linspace(40, 160, 500)
    density = np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    plt.figure(figsize=(7, 5))
    plt.plot(x, density, "b", linewidth=2, label="demand density D ~ N(100, 15²)")
    tail = x >= threshold
    plt.fill_between(x[tail], 0, density[tail], color="red", alpha=0.3, label="5% violation tail")
    plt.axvline(threshold, color="green", linestyle="--", label=f"stock s = {threshold:.1f}")
    plt.xlabel("demand D"); plt.ylabel("density")
    plt.title("Chance constraint  P(D ≤ s) ≥ 0.95")
    plt.legend(fontsize=8); plt.grid(True, alpha=0.3); plt.tight_layout()
    return "030702_chance_constraint_density"


chance_constraint_density()
plt.savefig('030702_chance_constraint_density.png', dpi=110)
