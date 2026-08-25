from typing import Tuple
import torch 
import numpy as np

# For DANN
def DANN_label_loss(label_pred: torch.Tensor, label_true: torch.Tensor) -> torch.Tensor:
    label_i = torch.where(label_true == 1)
    return torch.log(1 / label_pred[label_i]).sum()

def DANN_domain_loss(d_pred: float, d_true: int) -> torch.Tensor:
    return torch.Tensor(
        d_true * torch.log(1 / d_pred) + (1 - d_true) * torch.log(1 / (1 - d_pred))
    ).sum()

