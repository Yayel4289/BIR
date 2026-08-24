from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from exps.points_classification.utils.point import Point
from src.data.data_loader import get_loaders_from_tensors, tensor_train_val_split
from exps.points_classification.utils.domain import Domain

S = 0 # Source
T = 1 # Target


def generate_domains() -> List[Point]:
    x1 = np.arange(0, 10, 0.1)
    x2 = np.arange(0, 10, 0.1)
    mu1 = x1
    mu2 = 10 + 0*x2
    sigma1 = 1 + 0*x1
    sigma2 = 0.5 + 0.3*x2
    rho = 0.2

    domains = [
        Domain(500, (0, 3.5), S),
        Domain(2000, (6.5, 10), T)
    ]

    points = []
    for domain in domains:
        a, b = domain.interval
        indices = np.where((x1 >= a) & (x1 <= b))[0]
        for _ in range(domain.n):
            i = np.random.choice(indices)
            mean = [mu1[i], mu2[i]]
            cov_matrix = [
                [sigma1[i]**2, rho * sigma1[i] * sigma2[i]],
                [rho * sigma1[i] * sigma2[i], sigma2[i]**2]
            ]
            p_x1, p_x2 = np.random.multivariate_normal(mean=mean, cov=cov_matrix)
            label = int(p_x2 >= 10)
            points.append(Point(p_x1, p_x2, domain.domain_id, label))

    return points


def get_X_y_from_domain(points: List[Point], domain: int, shuffle: bool=True) -> Tuple[torch.Tensor, torch.Tensor]:
    filtered_pts = list(filter(lambda p: p.domain == domain, points))
    X_dom = np.asarray([[p.x1, p.x2] for p in filtered_pts])
    y_dom = np.asarray([[p.label, p.domain] for p in filtered_pts])
    if shuffle:
        shuffled_idxs = np.random.permutation(len(X_dom))
        X_dom, y_dom = X_dom[shuffled_idxs], y_dom[shuffled_idxs]

    X_dom_tensor = torch.tensor(X_dom, dtype=torch.float32)
    y_dom_tensor = torch.tensor(y_dom, dtype=torch.float32)
    return X_dom_tensor, y_dom_tensor


def get_X_y_percent_from_target(points: List[Point], percent=2) -> Tuple[torch.Tensor, torch.Tensor]:
    target_pts = list(filter(lambda p: p.domain == T, points))
    X_target = np.asarray([[p.x1, p.x2] for p in filtered_pts])
    y_target = np.asarray([[torch.nan, p.domain] for p in filtered_pts]) # NaN for absence of label in target 

    samples_nbr = int(len(X_target) * percent / 100)
    rand_select_idxs = np.random.choice(len(X_target), size=samples_nbr, replace=False)
    X_target = X_target[rand_select_idxs]
    y_target = y_target[rand_select_idxs]

    X_target_tensor = torch.tensor(X_target, dtype=torch.float32)
    y_target_tensor = torch.tensor(y_target, dtype=torch.float32)
    return X_target_tensor, y_target_tensor


def get_train_val_loaders_from_source(points: List[Point]) -> Tuple[DataLoader, DataLoader]:
    X_tensor, y_tensor = get_X_y_from_domain(points, S)
    X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor = tensor_train_val_split(X_tensor, y_tensor)
    return get_loaders_from_tensors(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)


def get_train_val_loaders_from_source_target(points: List[Point]) -> Tuple[DataLoader, DataLoader]:
    X_source_tensor, y_source_tensor = get_X_y_from_domain(points, S, shuffle=False)
    X_target_tensor, y_target_tensor = get_X_y_percent_from_target(points)
    X_tensor, y_tensor = torch.cat(X_source_tensor, X_target_tensor), torch.cat(y_source_tensor, y_target_tensor)
    shuffled_idxs = torch.randperm(len(X_tensor))
    X_tensor, y_tensor = X_tensor[shuffled_idxs], y_tensor[shuffled_idxs]
    X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor = tensor_train_val_split(X_tensor, y_tensor)
    return get_loaders_from_tensors(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)

