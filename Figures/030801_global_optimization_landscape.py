import numpy as np
import matplotlib.pyplot as plt


def global_optimization_landscape():
    x = np.linspace(-2.2, 3.2, 500)
    f = 3 * x**4 - 4 * x**3 - 12 * x**2
    plt.figure(figsize=(7, 5))
    plt.plot(x, f, linewidth=2)
    plt.scatter([2], [-32], color="green", marker="*", s=240, zorder=4, label="global minimum (2, -32)")
    plt.scatter([-1], [-5], color="red", s=70, zorder=3, label="local minimum (-1, -5)")
    plt.scatter([0], [0], color="orange", s=70, zorder=3, label="local maximum (0, 0)")
    plt.xlabel("$x$"); plt.ylabel("$f(x)$"); plt.title("Multimodal landscape: two minima, one global")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    return "030801_global_optimization_landscape"


global_optimization_landscape()
plt.savefig('030801_global_optimization_landscape.png', dpi=110)
