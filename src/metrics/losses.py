from typing import Tuple
import torch 
import numpy as np

# For DANN
def DANN_label_loss(label_pred: torch.Tensor, label_true: torch.Tensor) -> torch.Tensor:
    idx = label_true.view(-1, 1)
    true_class_prob = label_pred.gather(1, idx).clamp(min=1e-7)
    return torch.log(1 / true_class_prob).mean()

def DANN_domain_loss(d_pred: torch.Tensor, d_true: torch.Tensor) -> torch.Tensor:
    d_true = d_true.float().view(-1, 1)
    d_pred = d_pred.clamp(min=1e-7, max=1 - 1e-7)
    return (d_true * torch.log(1 / d_pred) + (1 - d_true) * torch.log(1 / (1 - d_pred))).mean()
