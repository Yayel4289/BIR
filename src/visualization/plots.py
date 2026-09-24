from typing import List
import matplotlib

from models.abstract_model import AbstractModel
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import torch
import numpy as np

from exps.points_classification.utils.point import Point

ALPHA = 0.5

def circle_handle(color, label, size=50, alpha=ALPHA):
    return plt.scatter([], [], color=color, s=size, alpha=alpha, label=label)

def get_pt_label(label: int, domain: int):
    pt_label = "B" if label else "A"
    pt_domain = "Source" if domain < 0.5 else "Target"
    return f"{pt_domain} domain, Class {pt_label}"

def plot_pts(points: List[Point], continuous: bool = False, show_target_label=True):
    plt.figure(figsize=(12, 8))
    source_colors = ["red", "blue"]
    target_colors = ["orange", "cyan"] if show_target_label else ["black"] * 2
    colors_label0 = plt.cm.colors.LinearSegmentedColormap.from_list("source_cmap", [source_colors[0], target_colors[0]])
    colors_label1 = plt.cm.colors.LinearSegmentedColormap.from_list("target_cmap", [source_colors[1], target_colors[1]])
    color_map = [colors_label0, colors_label1]
    norm = Normalize(0, 1)

    for p in points:
        plt.scatter(p.x1, p.x2, alpha=ALPHA, color=color_map[p.label](norm(p.domain)))

    custom_handles = [
        circle_handle(source_colors[0], get_pt_label(0, 0)),
        circle_handle(source_colors[1], get_pt_label(1, 0)),
    ]
    if show_target_label:
        custom_handles.extend([
            circle_handle(target_colors[0], get_pt_label(0, 1)),
            circle_handle(target_colors[1], get_pt_label(1, 1)),
        ])
    else:
        custom_handles.append(circle_handle("black", "Target domain"))

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(True)
    plt.legend(handles=custom_handles, loc="lower right")
    plt.axis("equal")

def plot_decision_boundary(points: List[Point], model: AbstractModel):
    X = np.asarray([[p.x1, p.x2] for p in points])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 500), np.linspace(x2_min, x2_max, 500))
    grid_points = np.c_[xx1.ravel(), xx2.ravel()]

    with torch.inference_mode():
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
        predictions = model.predict_probs(grid_tensor)
        predicted_labels = (predictions[:, 1] > predictions[:, 0]).float().numpy().reshape(xx1.shape)

    # decision boundary
    plt.contourf(xx1, xx2, predicted_labels, alpha=0.1, colors=["red", "blue"], levels=[0, 0.5, 1])
    plt.contour(xx1, xx2, predicted_labels, levels=[0.5], colors='black', linewidths=1.5)

def plot_image(images: torch.Tensor, index: int, normalized: bool = True) -> None:
    img = images[index]
    img = img.permute(1, 2, 0) # (C, H, W) -> (H, W, C)
    if normalized:
        img *= 255
    img = img.numpy().astype("uint8")

    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Image {index}")

def show():
    plt.show()
