from typing import Callable
import torch 
from torch import nn
from torch.utils.data import DataLoader
from models.dann.dann import DANN
from training.baseline_trainer import BaselineTrainer

class DANN_Trainer(BaselineTrainer):
    def __init__(
            self, 
            model: DANN, 
            source_train_loader: DataLoader, 
            source_val_loader: DataLoader, 
            target_train_loader: DataLoader,
            label_loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
            domain_loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            epochs: int = 100, 
            lr: float = 0.01
    ) -> None:
        if source_train_loader.batch_size != target_train_loader.batch_size: # Source/Target proportions : 50/50
            raise ValueError(
                f"Source and Target DataLoaders must have the same batch size ({source_train_loader.batch_size} and {target_train_loader.batch_size}"
            )
        super().__init__(model, source_train_loader, source_val_loader, label_loss_function, epochs=epochs, lr=lr)
        self.target_train_loader = target_train_loader
        self.domain_loss_function = domain_loss_function

    def _train_epoch(self) -> Tuple[float, float, float, float, float]: 
        self.model.train()
        train_loss = 0
        epoch_source_label_loss = 0
        epoch_source_domain_loss = 0
        epoch_target_domain_loss = 0
        epoch_domain_loss = 0
        for X_source_train_batch, y_source_train_batch in self.source_train_loader:
            X_target_train_batch, y_target_train_batch = next(iter(self.target_train_loader))

            source_labels_pred, source_domains_pred = self.model(X_source_train_batch)
            _, target_domains_pred = self.model.forward(X_target_train_batch, only_domain=True)
            source_train_batch_labels, source_train_batch_domains = y_source_train_batch[:, 0].view(-1, 1), y_train_batch[:, 1].view(-1, 1)
            target_train_batch_domains = y_target_train_batch[:, 1].view(-1, 1)

            source_label_loss = self.label_loss_function(source_labels_pred, source_train_batch_labels)
            source_domain_loss = self.domain_loss_function(source_domains_pred, source_train_batch_domains)
            target_domain_loss = self.domain_loss_function(target_domains_pred, target_train_batch_domains)
            domain_loss = source_domain_loss + target_domain_loss

            self.optimizer.zero_grad()
            source_label_loss.backward()
            domain_loss.backward()
            self.optimizer.step()
            
            epoch_source_label_loss += source_label_loss.item()
            epoch_source_domain_loss += source_domain_loss.item()
            epoch_target_domain_loss += target_domain_loss.item()
            epoch_domain_loss += domain_loss.item()

        train_loss = epoch_source_label_loss + epoch_domain_loss
        mean_train_loss = train_loss / len(self.source_train_loader)
        mean_source_label_loss = epoch_source_label_loss / len(self.source_train_loader)
        mean_domain_loss = epoch_domain_loss / len(self.source_train_loader)
        mean_source_domain_loss = epoch_source_domain_loss / len(self.source_train_loader)
        mean_target_domain_loss = epoch_target_domain_loss / len(self.source_train_loader)
        return mean_train_loss, mean_source_label_loss, mean_domain_loss, mean_source_domain_loss, mean_target_domain_loss

