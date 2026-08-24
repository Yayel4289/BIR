from typing import Callable
import torch 
from torch import nn
from torch.utils.data import DataLoader
from training.baseline_trainer import BaselineTrainer

class DANN_Trainer(BaselineTrainer):
    def __init__(
            self, 
            model: nn.Module, 
            train_loader: DataLoader, 
            test_loader: DataLoader, 
            label_loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
            domain_loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            epochs: int = 100, 
            lr: float = 0.01
    ) -> None:
        super().__init__(model, train_loader, test_loader, label_loss_function, epochs, lr)
        self.domain_loss_function = domain_loss_function

