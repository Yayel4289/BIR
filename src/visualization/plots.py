from typing import List
import matplotlib

from models.abstract_model import AbstractModel
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch
import numpy as np

from exps.points_classification.utils.point import Point

def plot_pts(points: List[Point], continuous: bool = False, show_target_label=True):
    plt.figure(figsize=(12, 8))
    source_colors = ["red", "blue"]
    target_colors = ["orange", "cyan"] if show_target_label else ["black"] * 2
    colors_label0 = plt.cm.colors.LinearSegmentedColormap.from_list("source_cmap", [source_colors[0], target_colors[0]])
    colors_label1 = plt.cm.colors.LinearSegmentedColormap.from_list("target_cmap", [source_colors[1], target_colors[1]])
    color_map = [colors_label0, colors_label1]
    norm = Normalize(0, 1)

    for p in points:
        plt.scatter(p.x1, p.x2, alpha=0.5, color=color_map[p.label](norm(p.domain)))

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(True)
    #plt.legend()
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
