import numpy as np
import matplotlib.pyplot as plt


def integer_program_feasible_lattice():
    x_1 = np.linspace(0, 4.5, 300)
    upper = np.minimum((24 - 6 * x_1) / 4, (6 - x_1) / 2)
    plt.figure(figsize=(7, 5))
    plt.fill_between(x_1, 0, upper, where=upper >= 0, alpha=0.2, label="LP feasible region")
    lattice = [(a, b) for a in range(5) for b in range(4) if 6 * a + 4 * b <= 24 and a + 2 * b <= 6]
    plt.scatter([p[0] for p in lattice], [p[1] for p in lattice], color="black", zorder=3, label="integer feasible points")
    plt.scatter([3], [1.5], color="red", marker="*", s=220, zorder=4, label="LP optimum (3, 1.5)")
    plt.scatter([4], [0], color="green", marker="s", s=130, zorder=4, label="integer optimum (4, 0)")
    plt.xlabel("$x_1$"); plt.ylabel("$x_2$"); plt.title("Integer program: feasible lattice vs LP relaxation")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    return "030601_integer_program_feasible_lattice"


integer_program_feasible_lattice()
plt.savefig('030601_integer_program_feasible_lattice.png', dpi=110)
