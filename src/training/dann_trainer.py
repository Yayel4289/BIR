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
            val_loader: DataLoader, 
            label_loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
            domain_loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            epochs: int = 100, 
            lr: float = 0.01
    ) -> None:
        super().__init__(model, train_loader, val_loader, label_loss_function, epochs, lr)
        self.domain_loss_function = domain_loss_function

    def _train_epoch(self) -> Tuple[float, float, float]: 
        self.model.train()
        train_loss = 0
        epoch_label_loss = 0
        epoch_domain_loss = 0
        for X_train_batch, y_train_batch in self.train_loader:
            labels_pred, domains_pred = self.model(X_train_batch)
            y_train_batch_labels, y_train_batch_domains = y_train_batch[:, 0].view(-1, 1), y_train_batch[:, 1].view(-1, 1)
            label_loss = self.label_loss_function(labels_pred, y_train_batch_labels)
            domain_loss = self.domain_loss_function(domains_pred, y_train_batch_domains)

            self.optimizer.zero_grad()
            label_loss.backward()
            domain_loss.backward()
            self.optimizer.step()
            
            epoch_label_loss += label_loss.item()
            epoch_domain_loss += domain_loss.item()

        train_loss = epoch_label_loss + epoch_domain_loss
        mean_train_loss = train_loss / len(self.train_loader)
        mean_label_loss = epoch_label_loss / len(self.train_loader)
        mean_domain_loss = epoch_domain_loss / len(self.train_loader)
        return mean_train_loss, mean_label_loss, mean_domain_loss

    def _train_epoch

