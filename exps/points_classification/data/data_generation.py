from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from exps.points_classification.utils.point import Point
from src.data.data_loader import get_loaders_from_tensors
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

def get_X_y_from_domain(points: List[Point], domain: int) -> Tuple[torch.Tensor, torch.Tensor]:
    filtered_pts = list(filter(lambda p: p.domain == domain, points))
    X_dom = np.asarray([[p.x1, p.x2] for p in filtered_pts])
    y_dom = np.asarray([[p.label, p.domain] for p in filtered_pts])

    shuffled_idx = np.random.permutation(len(X_dom))
    X_dom = X_dom[shuffled_idx]
    y_dom = y_dom[shuffled_idx]

    X_dom_tensor = torch.tensor(X_dom, dtype=torch.float32)
    y_dom_tensor = torch.tensor(y_dom, dtype=torch.float32)

    return X_dom_tensor, y_dom_tensor

def get_train_test_loaders_from_domains(points: List[Point]) -> Tuple[DataLoader, DataLoader]:
    X_train_tensor, y_train_tensor = get_X_y_from_domain(points, S)
    X_test_tensor, y_test_tensor = get_X_y_from_domain(points, T)
    return get_loaders_from_tensors(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)

