from typing import Tuple
import torch 
from torch.utils.data import TensorDataset, DataLoader

from src.data.cycling_sampler import CyclingSampler

def tensor_train_val_split(
        X_tensor: torch.Tensor, 
        y_tensor: torch.Tensor, 
        train_percent: int=75
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(X_tensor) != len(y_tensor): 
        raise ValueError(f"X_tensor and y_tensor must be the same length ({len(X_tensor)} and {len(y_tensor)} resp.)")
    train_row_nbr = int(train_percent / 100 * len(X_tensor))
    val_row_nbr = len(X_tensor) - train_row_nbr
    X_train_tensor, X_val_tensor = X_tensor.split([train_row_nbr, val_row_nbr])
    y_train_tensor, y_val_tensor = y_tensor.split([train_row_nbr, val_row_nbr])
    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor

def tensor_train_val_test_split(
        X_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        train_percent: int = 70,
        val_percent: int = 20
        # test_percent = 10
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(X_tensor) != len(y_tensor):
        raise ValueError(f"X_tensor and y_tensor must be the same length ({len(X_tensor)} and {len(y_tensor)} resp.)")
    train_row_nbr = int(train_percent / 100 * len(X_tensor))
    val_row_nbr = int(val_percent / 100 * len(X_tensor))
    test_row_nbr = len(X_tensor) - train_row_nbr - val_row_nbr
    X_train_tensor, X_val_tensor, X_test_tensor = X_tensor.split([train_row_nbr, val_row_nbr, test_row_nbr])
    y_train_tensor, y_val_tensor, y_test_tensor = y_tensor.split([train_row_nbr, val_row_nbr, test_row_nbr])
    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor
    
###

def get_loader_from_tensors(X: torch.Tensor, y: torch.Tensor, batch_size: int = 32, test: bool = False) -> DataLoader:
    dataset = TensorDataset(X, y)
    sampler = CyclingSampler(dataset, batch_size=batch_size) if not test else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
    )
    return loader

def get_train_val_loaders_from_tensors(
        X_train_tensor: torch.Tensor, y_train_tensor: torch.Tensor,
        X_val_tensor: torch.Tensor, y_val_tensor: torch.Tensor,
        batch_size: int=32
        ) -> Tuple[DataLoader]:
    train_loader = get_loader_from_tensors(X_train_tensor, y_train_tensor, batch_size=batch_size)
    val_loader = get_loader_from_tensors(X_val_tensor, y_val_tensor, batch_size=batch_size)
    return train_loader, val_loader

def get_train_val_test_loaders_from_tensors(
        X_train_tensor: torch.Tensor, y_train_tensor: torch.Tensor,
        X_val_tensor: torch.Tensor, y_val_tensor: torch.Tensor,
        X_test_tensor: torch.Tensor, y_test_tensor: torch.Tensor,
        batch_size: int=32
        ) -> Tuple[DataLoader]:
    train_loader, val_loader = get_train_val_loaders_from_tensors(
        X_train_tensor, y_train_tensor,
        X_val_tensor, y_val_tensor,
        batch_size=batch_size
    )
    test_loader = get_loader_from_tensors(X_test_tensor, y_test_tensor, batch_size=batch_size, test=True)
    return train_loader, val_loader, test_loader


def get_source_loaders_from_tensors(
        X_source_train_tensor: torch.Tensor, y_source_train_tensor: torch.Tensor,
        X_source_val_tensor: torch.Tensor, y_source_val_tensor: torch.Tensor,
        batch_size: int=32
        ) -> Tuple[DataLoader, DataLoader]:
    return get_train_val_loaders_from_tensors(
        X_source_train_tensor, y_source_train_tensor,
        X_source_val_tensor, y_source_val_tensor,
        batch_size=batch_size
    )

def get_source_train_val_test_loaders_from_tensors(
        X_source_train_tensor: torch.Tensor, y_source_train_tensor: torch.Tensor,
        X_source_val_tensor: torch.Tensor, y_source_val_tensor: torch.Tensor,
        X_source_test_tensor: torch.Tensor, y_source_test_tensor: torch.Tensor,
        batch_size: int=32
        ) -> Tuple[DataLoader, DataLoader]:
    return get_train_val_test_loaders_from_tensors(
        X_source_train_tensor, y_source_train_tensor,
        X_source_val_tensor, y_source_val_tensor,
        X_source_test_tensor, y_source_test_tensor,
        batch_size=batch_size
    )

def get_target_loaders_from_tensors(
        X_target_train_tensor: torch.Tensor, y_target_train_tensor: torch.Tensor,
        X_target_val_tensor: torch.Tensor, y_target_val_tensor: torch.Tensor,
        batch_size: int=32
        ) -> Tuple[DataLoader, DataLoader]:
    return get_train_val_loaders_from_tensors(
        X_target_train_tensor, y_target_train_tensor,
        X_target_val_tensor, y_target_val_tensor,
        batch_size=batch_size
    )   

def get_target_train_val_test_loaders_from_tensors(
        X_target_train_tensor: torch.Tensor, y_target_train_tensor: torch.Tensor,
        X_target_val_tensor: torch.Tensor, y_target_val_tensor: torch.Tensor,
        X_target_test_tensor: torch.Tensor, y_target_test_tensor: torch.Tensor,
        batch_size: int=32
        ) -> Tuple[DataLoader, DataLoader]:
    return get_train_val_test_loaders_from_tensors(
        X_target_train_tensor, y_target_train_tensor,
        X_target_val_tensor, y_target_val_tensor,
        X_target_test_tensor, y_target_test_tensor,
        batch_size=batch_size
    )

def get_source_target_loaders_from_tensors(
        X_source_tensor: torch.Tensor, y_source_tensor: torch.Tensor,
        X_target_tensor: torch.Tensor, y_target_tensor: torch.Tensor,
        batch_size=32
        ) -> Tuple[DataLoader, DataLoader]:
    source_loader = get_loader_from_tensors(X_source_tensor, y_source_tensor, batch_size=batch_size)
    target_loader = get_loader_from_tensors(X_target_tensor, y_target_tensor, batch_size=batch_size)
    return source_loader, target_loader


def get_source_target_train_val_loaders_from_tensors(
        X_source_tensor: torch.Tensor, y_source_tensor: torch.Tensor,
        X_target_tensor: torch.Tensor, y_target_tensor: torch.Tensor,
        batch_size=32
        ) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    X_source_train_tensor, y_source_train_tensor, X_source_val_tensor, y_source_val_tensor = tensor_train_val_split(X_source_tensor, y_source_tensor)
    source_train_loader, source_val_loader = get_source_loaders_from_tensors(
        X_source_train_tensor, y_source_train_tensor, 
        X_source_val_tensor, y_source_val_tensor,
        batch_size=batch_size
    )

    X_target_train_tensor, y_target_train_tensor, X_target_val_tensor, y_target_val_tensor = tensor_train_val_split(X_target_tensor, y_target_tensor)
    target_train_loader, target_val_loader = get_target_loaders_from_tensors(
        X_target_train_tensor, y_target_train_tensor,
        X_target_val_tensor, y_target_val_tensor,
        batch_size=batch_size
    )

    return source_train_loader, source_val_loader, target_train_loader, target_val_loader

def get_source_target_train_val_test_loaders_from_tensor(
        X_source_tensor: torch.Tensor, y_source_tensor: torch.Tensor,
        X_target_tensor: torch.Tensor, y_target_tensor: torch.Tensor,
        batch_size=32
        ) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    source_train_loader, source_val_loader, source_test_loader = get_source_train_val_test_loaders_from_tensors(
        *tensor_train_val_test_split(X_source_tensor, y_source_tensor)
    )
    target_train_loader, target_val_loader, target_test_loader = get_target_train_val_test_loaders_from_tensors(
        *tensor_train_val_test_split(X_target_tensor, y_target_tensor)
    )
    return source_train_loader, target_train_loader, source_val_loader, target_val_loader, source_test_loader, target_test_loader


