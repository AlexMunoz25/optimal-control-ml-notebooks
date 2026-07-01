import numpy as np
import matplotlib.pyplot as plt


def branch_and_bound_tree():
    plt.figure(figsize=(7, 5))
    nodes = {"root": (0.5, 1.0, "LP=21.0"),
             "L": (0.25, 0.6, "x1<=3\nLP=20.6"), "R": (0.75, 0.6, "x1>=4\nLP=20.0"),
             "LL": (0.12, 0.2, "x2<=1\nint=19"), "LR": (0.38, 0.2, "x2>=2\ninfeas"),
             "RR": (0.75, 0.2, "int=20*")}
    edges = [("root", "L"), ("root", "R"), ("L", "LL"), ("L", "LR"), ("R", "RR")]
    for a, b in edges:
        plt.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], "k-", zorder=1)
    for key, (px, py, label) in nodes.items():
        color = "lightgreen" if "*" in label else ("lightgray" if "infeas" in label else "lightblue")
        plt.scatter([px], [py], s=2600, color=color, edgecolors="black", zorder=2)
        plt.text(px, py, label, ha="center", va="center", fontsize=8, zorder=3)
    plt.axis("off"); plt.title("Branch and bound: incumbent (4, 0) with value 20")
    plt.tight_layout()
    return "030802_branch_and_bound_tree"


branch_and_bound_tree()
plt.savefig('030802_branch_and_bound_tree.png', dpi=110)
