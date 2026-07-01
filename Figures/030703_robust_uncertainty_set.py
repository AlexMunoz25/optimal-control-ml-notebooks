import numpy as np
import matplotlib.pyplot as plt


def robust_uncertainty_set():
    plt.figure(figsize=(6, 6))
    nominal = np.array([1.0, 1.0])
    half = 0.2
    box = plt.Rectangle(nominal - half, 2 * half, 2 * half, alpha=0.2, color="blue", label="box uncertainty set")
    plt.gca().add_patch(box)
    theta = np.linspace(0, 2 * np.pi, 200)
    plt.plot(nominal[0] + half * np.cos(theta), nominal[1] + half * np.sin(theta), "b--", alpha=0.6, label="ellipsoidal set")
    plt.scatter(*nominal, color="black", zorder=3, label="nominal ā")
    plt.scatter(nominal[0] + half, nominal[1] + half, color="red", zorder=3, label="worst-case a")
    plt.xlabel("a₁"); plt.ylabel("a₂"); plt.title("Robust optimization: uncertainty set")
    plt.legend(fontsize=8); plt.grid(True, alpha=0.3); plt.axis("equal"); plt.tight_layout()
    return "030703_robust_uncertainty_set"


robust_uncertainty_set()
plt.savefig('030703_robust_uncertainty_set.png', dpi=110)
