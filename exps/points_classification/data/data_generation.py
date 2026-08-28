from typing import Callable, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from exps.points_classification.utils.point import Point
from src.data.data_loader import get_source_loaders_from_tensors, get_source_target_loaders_from_tensors, tensor_train_val_split
from exps.points_classification.utils.domain import Domain

S = 0 # Source
T = 1 # Target

def high_freq_sin_boundary_cond(p_x1: float, p_x2: float) -> bool:
    return p_x2 >= 10+np.sin(p_x1*2)

def high_ampl_sin_boundary_cond(p_x1: float, p_x2: float) -> bool:
    return p_x2 >= 10+1*np.sin(p_x1)

def get_point_label(p_x1: float, p_x2: float, condition_func: Callable[float, float] = high_freq_sin_boundary_cond) -> int:
    return int(condition_func(p_x1, p_x2))

def generate_split_domains() -> List[Point]:
    x1 = np.arange(0, 10, 0.1)
    x2 = np.arange(0, 10, 0.1)
    mu1 = x1
    mu2 = 10 + 0*x2
    sigma1 = 1 + 0*x1
    sigma2 = 0.5 + 0.3*x2
    rho = 0.2

    domains = [
        Domain(S, 2000, (0, 3.5)),
        Domain(T, 1000, (6.5, 10))
    ]
    return generate_normal_distr_points(domains, x1, x2, mu1, mu2, sigma1, sigma2, rho)


def generate_fused_domains() -> List[Point]:
    x1 = np.arange(0, 10, 0.1)
    x2 = np.arange(0, 10, 0.1)

    domains = [
        Domain(S, 500, (0, 20), (0, 20)),
        Domain(T, 500, (14, 19), (7, 12))
    ]
    return generate_circ_uniform_points(domains, x1, x2)


def generate_circ_uniform_points(domains: List[Domain], x1: np.ndarray, x2: np.ndarray):
    points = []
    for domain in domains:
        x1_min, x1_max = domain.x_interval
        x2_min, x2_max = domain.y_interval

        # Ellipsis details
        center_x1 = (x1_min + x1_max) / 2
        center_x2 = (x2_min + x2_max) / 2
        horiz_radius = (x1_max - x1_min) / 2
        vert_radius = (x2_max - x2_min) / 2
        for _ in range(domain.n):
            theta = np.random.uniform(0, 2*np.pi)
            base_radius = np.sqrt(np.random.random())
            p_x1 = center_x1 + base_radius * horiz_radius * np.cos(theta)
            p_x2 = center_x2 + base_radius * vert_radius * np.sin(theta)
            #if 9 <= p_x2 <= 11: 
            #    p_x2 += np.random.choice((-1, 1)) * (1 + np.random.random())
            label = get_point_label(p_x1, p_x2, high_ampl_sin_boundary_cond)
            points.append(Point(p_x1, p_x2, domain.domain_id, label))
    return points
        

def generate_normal_distr_points(
        domains: List[Domain], 
        x1: np.ndarray, 
        x2: np.ndarray, 
        mu1: np.ndarray, 
        mu2: np.ndarray, 
        sigma1: np.ndarray, 
        sigma2: np.ndarray,
        rho: float
        ) -> List[Point]:
    points = []
    for domain in domains:
        a, b = domain.x_interval
        indices = np.where((x1 >= a) & (x1 <= b))[0]
        for _ in range(domain.n):
            i = np.random.choice(indices)
            mean = [mu1[i], mu2[i]]
            cov_matrix = [
                [sigma1[i]**2, rho * sigma1[i] * sigma2[i]],
                [rho * sigma1[i] * sigma2[i], sigma2[i]**2]
            ]
            p_x1, p_x2 = np.random.multivariate_normal(mean=mean, cov=cov_matrix)
            label = get_point_label(p_x1, p_x2)
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

def get_X_from_domain(points: List[Point], domain: int) -> torch.Tensor:
    filtered_pts = list(filter(lambda p: p.domain == domain, points))
    X_dom = np.asarray([[p.x1, p.x2] for p in filtered_pts])
    X_dom_tensor = torch.tensor(X_dom, dtype=torch.float32)
    return X_dom_tensor

def get_X_from_source_target(points: List[Point]) -> Tuple[torch.Tensor]:
    X_source_tensor = get_X_from_domain(points, S)
    X_target_tensor = get_X_from_domain(points, T)
    return X_source_tensor, X_target_tensor

def get_X_y_percent_from_target(points: List[Point], percent=10) -> Tuple[torch.Tensor, torch.Tensor]:
    target_pts = list(filter(lambda p: p.domain == T, points))
    X_target = np.asarray([[p.x1, p.x2] for p in target_pts])
    y_target = np.asarray([[torch.nan, p.domain] for p in target_pts]) # NaN for absence of label in target 

    samples_nbr = int(len(X_target) * percent / 100)
    rand_select_idxs = np.random.choice(len(X_target), size=samples_nbr, replace=False)
    X_target = X_target[rand_select_idxs]
    y_target = y_target[rand_select_idxs]

    X_target_tensor = torch.tensor(X_target, dtype=torch.float32)
    y_target_tensor = torch.tensor(y_target, dtype=torch.float32)
    return X_target_tensor, y_target_tensor


def get_train_val_loaders_from_source(points: List[Point]) -> Tuple[DataLoader, DataLoader]:
    X_source_tensor, y_source_tensor = get_X_y_from_domain(points, S)
    X_source_train_tensor, y_source_train_tensor, X_source_val_tensor, y_source_val_tensor = tensor_train_val_split(X_source_tensor, y_source_tensor)
    return get_source_loaders_from_tensors(X_source_train_tensor, y_source_train_tensor, X_source_val_tensor, y_source_val_tensor)

def get_train_val_loaders_from_source_target(points: List[Point]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    X_source_tensor, y_source_tensor = get_X_y_from_domain(points, S, shuffle=False)
    X_target_train_tensor, y_target_train_tensor = get_X_y_percent_from_target(points)
    return get_source_target_loaders_from_tensors(X_source_tensor, y_source_tensor, X_target_train_tensor, y_target_train_tensor)

