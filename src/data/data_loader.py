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

def get_loaders_from_tensors(
    X_train_tensor: torch.Tensor, y_train_tensor: torch.Tensor,
    X_val_tensor: torch.Tensor, y_val_tensor: torch.Tensor,
    batch_size=32
) -> Tuple[DataLoader, DataLoader]:

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader

