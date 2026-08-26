from typing import Tuple
import torch 
import numpy as np

# For DANN
def DANN_label_loss(label_pred: torch.Tensor, label_true: torch.Tensor) -> torch.Tensor:
    idx = label_true.view(-1, 1)
    true_class_prob = label_pred.gather(1, idx)
    return torch.log(1 / true_class_prob).mean()

def DANN_domain_loss(d_pred: float, d_true: int) -> torch.Tensor:
    return (d_true * torch.log(1 / d_pred) + (1 - d_true) * torch.log(1 / (1 - d_pred))).mean()

