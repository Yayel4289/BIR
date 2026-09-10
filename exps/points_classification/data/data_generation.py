from typing import Callable, Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.data_loader_set import DataLoaderSet
from exps.points_classification.utils.moon_domain import MoonDomain
from exps.points_classification.utils.point import Point
from src.data.data_loader import get_source_loaders_from_tensors, get_source_target_loaders_from_tensors, get_source_target_train_val_loaders_from_tensors, tensor_train_val_split
from exps.points_classification.utils.domain import Domain
from sklearn.datasets import make_moons

S = 0 # Source
T = 1 # Target

CID_BOUND = 0.5

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

def generate_intertwinning_moons(continuous: bool = False, noise: float = 0.075, tilt: float = np.pi / 5, n_by_dom: int = 300) -> List[Point]:
    if not continuous:
        return _generate_intertwinning_moons_discrete(noise, tilt, n_by_dom)
    else:
        return _generate_intertwinning_moons_continuous(noise, tilt, n_by_dom*2)

def _generate_intertwinning_moons_discrete(noise: float, tilt: float, n_by_dom: int) -> List[Point]:
    domains = [
        MoonDomain(S, n_by_dom, 0),
        MoonDomain(T, n_by_dom, tilt)
    ]

    pts = []
    for domain in domains:
        X_dom, y_dom = make_moons(domain.n, noise=noise)
        X_dom = rotate((0.5, 0.25), X_dom, domain.tilt_rad)
        domain_pts = [Point(X_dom[i, 0], X_dom[i, 1], domain.domain_id, y_dom[i]) for i in range(domain.n)]
        pts.extend(domain_pts)
    return pts

def _generate_intertwinning_moons_continuous(noise: float, tilt_max: float, n_total: int, tilt_mult: float = 1.5) -> List[Point]:
    X, y = make_moons(n_total, noise=noise)
    t = np.random.uniform(S, T, size=n_total)
    angles = t * tilt_max * tilt_mult # solves double noises diff
    X_rot = rotate((0.5, 0.25), X, angles)
    return [Point(X_rot[i, 0], X_rot[i, 1], t[i], y[i]) for i in range(n_total)]        

def rotate(origin: Tuple[float, float], pts:np.ndarray, angle: [float | np.ndarray]):
    o = np.atleast_2d(origin)
    d = pts - o
    cos = np.cos(angle)
    sin = np.sin(angle)
    x_rot = d[:, 0] * cos - d[:, 1] * sin
    y_rot = d[:, 0] * sin + d[:, 1] * cos
    return np.stack([x_rot, y_rot], axis=1) + o


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


def get_pts_from_domain(pts: List[Point], domain: int) -> List[Point]:
    return list(filter(lambda p: np.abs(domain - p.domain) < np.abs(domain - CID_BOUND), pts))


def get_X_y_from_domain(points: List[Point], domain: int, shuffle: bool=True) -> Tuple[torch.Tensor, torch.Tensor]:
    filtered_pts = get_pts_from_domain(points, domain)
    X_dom = np.asarray([[p.x1, p.x2] for p in filtered_pts])
    y_dom = np.asarray([[p.label, p.domain] for p in filtered_pts])
    if shuffle:
        shuffled_idxs = np.random.permutation(len(X_dom))
        X_dom, y_dom = X_dom[shuffled_idxs], y_dom[shuffled_idxs]

    X_dom_tensor = torch.tensor(X_dom, dtype=torch.float32)
    y_dom_tensor = torch.tensor(y_dom, dtype=torch.float32)
    return X_dom_tensor, y_dom_tensor

def get_X_y_labels_from_source_target(points: List[Point]) -> Tuple[torch.Tensor]:
    X_source_tensor, y_source_tensor = get_X_y_from_domain(points, S, shuffle=False)
    X_target_tensor, y_target_tensor = get_X_y_from_domain(points, T, shuffle=False)
    X_tensor = torch.cat((X_source_tensor, X_target_tensor))
    y_tensor = torch.cat((y_source_tensor, y_target_tensor))
    return X_tensor, y_tensor[:, 0]

def get_X_y_from_source_target(points: List[Point]) -> Tuple[torch.Tensor, torch.Tensor,torch.Tensor, torch.Tensor]:
    X_source_tensor, y_source_tensor = get_X_y_from_domain(points, S, shuffle=False)
    X_target_tensor, y_target_tensor = get_X_y_from_domain(points, T, shuffle=False)
    return X_source_tensor, y_source_tensor, X_target_tensor, y_target_tensor

def get_X_y_labels_from_target(points: List[Point]) -> Tuple[torch.Tensor]:
    X_target_tensor, y_target_tensor = get_X_y_from_domain(points, T, shuffle=False)
    return X_target_tensor, y_target_tensor[:, 0]

def get_X_y_labels_from_source(points: List[Point]) -> Tuple[torch.Tensor]:
    X_source_tensor, y_source_tensor = get_X_y_from_domain(points, S, shuffle=False)
    return X_source_tensor, y_source_tensor[:, 0]

def get_X_from_domain(points: List[Point], domain: int) -> torch.Tensor:
    filtered_pts = get_pts_from_domain(points, domain)
    X_dom = np.asarray([[p.x1, p.x2] for p in filtered_pts])
    X_dom_tensor = torch.tensor(X_dom, dtype=torch.float32)
    return X_dom_tensor

def get_X_from_source_target(points: List[Point]) -> Tuple[torch.Tensor]:
    X_source_tensor = get_X_from_domain(points, S)
    X_target_tensor = get_X_from_domain(points, T)
    return X_source_tensor, X_target_tensor

def get_X_y_percent_from_target(points: List[Point], percent=10) -> Tuple[torch.Tensor, torch.Tensor]:
    target_pts = get_pts_from_domain(points, T)
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

def get_train_val_loaders_from_source_target(points: List[Point]) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    X_source_tensor, y_source_tensor = get_X_y_from_domain(points, S, shuffle=False)
    X_target_tensor, y_target_tensor = get_X_y_percent_from_target(points)
    return get_source_target_train_val_loaders_from_tensors(X_source_tensor, y_source_tensor, X_target_tensor, y_target_tensor)

def get_source_target_loaders_from_points(points: List[Point]) -> Tuple[DataLoader]:
    X_source_tensor, y_source_tensor = get_X_y_from_domain(points, S, shuffle=False)
    X_target_tensor, y_target_tensor = get_X_y_from_domain(points, T, shuffle=False)
    return get_source_target_loaders_from_tensors(X_source_tensor, y_source_tensor, X_target_tensor, y_target_tensor)

def get_moons_data_loader_set(continuous: bool = False) -> DataLoaderSet:
    train_val_pts = generate_intertwinning_moons(continuous=continuous)
    test_pts = generate_intertwinning_moons(continuous=continuous)
    source_train_loader, source_val_loader, target_train_loader, target_val_loader = get_train_val_loaders_from_source_target(train_val_pts)
    source_test_loader, target_test_loader = get_source_target_loaders_from_points(test_pts)
    return DataLoaderSet(source_train_loader, source_val_loader, source_test_loader, target_train_loader, target_val_loader, target_test_loader,test_pts)

