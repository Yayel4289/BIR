from typing import Tuple
import torch 
from torch.utils.data import TensorDataset, DataLoader

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

def get_source_loaders_from_tensors(
        X_source_train_tensor: torch.Tensor, y_source_train_tensor: torch.Tensor,
        X_source_val_tensor: torch.Tensor, y_source_val_tensor: torch.Tensor,
        batch_size=32
        ) -> Tuple[DataLoader, DataLoader]:
    source_train_dataset = TensorDataset(X_source_train_tensor, y_source_train_tensor)
    source_val_dataset = TensorDataset(X_source_val_tensor, y_source_val_tensor)

    source_train_loader = DataLoader(
        source_train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    source_val_loader = DataLoader(
        source_val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return source_train_loader, source_val_loader

def get_source_target_loaders_from_tensors(
        X_source_tensor: torch.Tensor, y_source_tensor: torch.Tensor,
        X_target_train_tensor: torch.Tensor, y_target_train_tensor: torch.Tensor,
        target_sampler,
        batch_size=32
        ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    X_source_train_tensor, y_source_train_tensor, X_source_val_tensor, y_source_val_tensor = tensor_train_val_split(X_source_tensor, y_source_tensor)
    source_train_loader, source_val_loader = get_source_loaders_from_tensors(
        X_source_train_tensor, y_source_train_tensor, 
        X_source_val_tensor, y_source_val_tensor
    )
    target_train_dataset = TensorDataset(X_target_train_tensor, y_target_train_tensor)
    target_train_loader = DataLoader(
        target_train_dataset,
        batch_size=batch_size,
        sampler=target_sampler,
        shuffle=True
    )
    return source_train_loader, source_val_loader, target_train_loader

