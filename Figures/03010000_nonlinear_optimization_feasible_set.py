from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def objective_surface(first_coordinate, second_coordinate):
    return (
        0.65 * (first_coordinate + 0.30) ** 2
        + 0.95 * (second_coordinate - 0.05) ** 2
    )


def save_current_view(key_press_event):
    if key_press_event.key == "j":
        figure.savefig(figure_path, bbox_inches="tight", pad_inches=0.08)
        print(f"saved {figure_path}")


figure_path = Path(__file__).with_suffix(".jpeg")

first_coordinate_values = np.linspace(-1.15, 1.05, 160)
second_coordinate_values = np.linspace(-1.05, 1.15, 160)
first_coordinate_grid, second_coordinate_grid = np.meshgrid(
    first_coordinate_values,
    second_coordinate_values,
)
objective_values = objective_surface(first_coordinate_grid, second_coordinate_grid)

angle_values = np.linspace(0.0, 2.0 * np.pi, 80)
radial_values = np.linspace(0.0, 1.0, 28)
angle_grid, radial_grid = np.meshgrid(angle_values, radial_values)

ellipse_angle = np.deg2rad(24.0)
cosine_angle = np.cos(ellipse_angle)
sine_angle = np.sin(ellipse_angle)
ellipse_first_axis = 0.58
ellipse_second_axis = 0.34
ellipse_center_first_coordinate = 0.40
ellipse_center_second_coordinate = 0.25

first_coordinate_offset = ellipse_first_axis * radial_grid * np.cos(angle_grid)
second_coordinate_offset = ellipse_second_axis * radial_grid * np.sin(angle_grid)
rotated_first_coordinate_offset = (
    cosine_angle * first_coordinate_offset
    - sine_angle * second_coordinate_offset
)
rotated_second_coordinate_offset = (
    sine_angle * first_coordinate_offset
    + cosine_angle * second_coordinate_offset
)
feasible_first_coordinates = (
    ellipse_center_first_coordinate + rotated_first_coordinate_offset
)
feasible_second_coordinates = (
    ellipse_center_second_coordinate + rotated_second_coordinate_offset
)
feasible_objective_values = objective_surface(
    feasible_first_coordinates,
    feasible_second_coordinates,
)
feasible_floor_values = np.zeros_like(feasible_objective_values)

translated_first_coordinate_grid = (
    first_coordinate_grid - ellipse_center_first_coordinate
)
translated_second_coordinate_grid = (
    second_coordinate_grid - ellipse_center_second_coordinate
)
aligned_first_coordinate_grid = (
    cosine_angle * translated_first_coordinate_grid
    + sine_angle * translated_second_coordinate_grid
)
aligned_second_coordinate_grid = (
    -sine_angle * translated_first_coordinate_grid
    + cosine_angle * translated_second_coordinate_grid
)
feasible_surface_mask = (
    (aligned_first_coordinate_grid / ellipse_first_axis) ** 2
    + (aligned_second_coordinate_grid / ellipse_second_axis) ** 2
    <= 1.04
)
visible_objective_values = objective_values.copy()
visible_objective_values[feasible_surface_mask] = np.nan

boundary_first_coordinate_offset = ellipse_first_axis * np.cos(angle_values)
boundary_second_coordinate_offset = ellipse_second_axis * np.sin(angle_values)
boundary_first_coordinates = ellipse_center_first_coordinate + (
    cosine_angle * boundary_first_coordinate_offset
    - sine_angle * boundary_second_coordinate_offset
)
boundary_second_coordinates = ellipse_center_second_coordinate + (
    sine_angle * boundary_first_coordinate_offset
    + cosine_angle * boundary_second_coordinate_offset
)
boundary_objective_values = objective_surface(
    boundary_first_coordinates,
    boundary_second_coordinates,
)

blue_surface_colormap = LinearSegmentedColormap.from_list(
    "blue_surface",
    ["#202184", "#4651c4", "#8e96ff", "#f6f7ff"],
)

figure = plt.figure(figsize=(8.6, 7.2), dpi=160)
axis = figure.add_subplot(111, projection="3d")

axis.plot_surface(
    first_coordinate_grid,
    second_coordinate_grid,
    visible_objective_values,
    cmap=blue_surface_colormap,
    linewidth=0,
    antialiased=True,
    alpha=0.70,
)
axis.plot_surface(
    feasible_first_coordinates,
    feasible_second_coordinates,
    feasible_floor_values,
    color="#0826ff",
    edgecolor="#0014b8",
    linewidth=0.18,
    alpha=0.96,
    shade=False,
)
axis.plot_surface(
    feasible_first_coordinates,
    feasible_second_coordinates,
    feasible_objective_values,
    color="#fff000",
    edgecolor="#333333",
    linewidth=0.28,
    alpha=0.88,
    shade=False,
)
axis.plot(
    boundary_first_coordinates,
    boundary_second_coordinates,
    boundary_objective_values,
    color="#fff000",
    linewidth=4.0,
)
axis.set_xlim(-1.15, 1.05)
axis.set_ylim(1.15, -1.05)
axis.set_zlim(0.0, 2.5)
axis.set_xlabel(r"$x_2$", labelpad=16, fontsize=18)
axis.set_ylabel(r"$x_1$", labelpad=14, fontsize=18)
axis.set_zlabel(r"$f(x_1,x_2)$", labelpad=16, fontsize=18)
axis.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
axis.set_yticks([-1.0, 0.0, 1.0])
axis.set_zticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
axis.view_init(elev=18, azim=-78)
axis.set_box_aspect((1.15, 1.15, 1.0))
axis.grid(True)

axis_panes = [axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane]
for axis_pane in axis_panes:
    axis_pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    axis_pane.set_edgecolor("#d6d6d6")

grid_axes = [axis.xaxis, axis.yaxis, axis.zaxis]
for grid_axis in grid_axes:
    grid_axis._axinfo["grid"]["color"] = (0.86, 0.86, 0.86, 0.65)

figure.tight_layout()
figure.canvas.mpl_connect("key_press_event", save_current_view)

print("Drag to rotate. Press j to save the current view as a JPEG.")
print(f"Output path: {figure_path}")

plt.show()
