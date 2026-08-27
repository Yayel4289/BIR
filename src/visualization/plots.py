from typing import List
import matplotlib

from models.abstract_model import AbstractModel
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import torch
import numpy as np

from exps.points_classification.utils.point import Point

def plot_pts(points: List[Point], show_target_label=True):
    plt.figure(figsize=(12, 8))
    target_colors = ["orange", "cyan"] if show_target_label else ["black"] * 2
    color_map = [
        ["red", "blue"], # source
        target_colors # target
    ]

    for p in points:
        plt.scatter(p.x1, p.x2, alpha=0.5, color=color_map[p.domain][p.label])
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
        predictions = model.predict(grid_tensor)
        predicted_labels = (predictions[:, 1] > predictions[:, 0]).float().numpy().reshape(xx1.shape)

    # decision boundary
    plt.contourf(xx1, xx2, predicted_labels, alpha=0.1, colors=["red", "blue"], levels=[0, 0.5, 1])
    plt.contour(xx1, xx2, predicted_labels, levels=[0.5], colors='black', linewidths=1.5)

def show():
    plt.show()

