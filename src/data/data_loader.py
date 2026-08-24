from typing import Tuple
import torch 
from torch.utils.data import TensorDataset, DataLoader

def get_loaders_from_tensors(
    X_train_tensor: torch.Tensor, y_train_tensor: torch.Tensor,
    X_test_tensor: torch.Tensor, y_test_tensor: torch.Tensor,
    batch_size=32
) -> Tuple[DataLoader, DataLoader]:

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader

