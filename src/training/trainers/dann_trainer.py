from typing import Callable
import torch 
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from models.dann.dann import DANN
from src.training.trainers.baseline_trainer import BaselineTrainer

class DANN_Trainer(BaselineTrainer):
    def __init__(
            self, 
            model: DANN, 
            source_train_loader: DataLoader, 
            source_val_loader: DataLoader, 
            target_train_loader: DataLoader,
            target_val_loader: DataLoader,
            label_loss_function: Callable[torch.Tensor, torch.Tensor], 
            domain_loss_function: Callable[torch.Tensor, torch.Tensor],
            epochs: int = 1000, 
            lr: float = 0.01,
            gamma: int = 10,
            max_lambda: int = 1,
            verbose: bool = True
    ) -> None:
        super().__init__(model, source_train_loader, source_val_loader, label_loss_function, epochs=epochs, lr=lr)
        self.target_train_loader = target_train_loader
        self.target_val_loader = target_val_loader
        self.domain_loss_function = domain_loss_function
        self.gamma = gamma
        self.max_lambda = max_lambda
        self.grl_lambda = 0
        self.verbose = verbose


    def log(self, *values: object, sep: str = " ") -> None:
        if self.verbose:
            print(*values, sep=sep)

    def compute_grl_lambda(self) -> float:
        p = self.epoch / self.epochs
        grl_lambda = self.max_lambda * (2 / (1 + np.exp(-self.gamma * p)) - 1)
        return grl_lambda

    def update_grl_lambda(self) -> None:
        self.grl_lambda = self.compute_grl_lambda()


    def _train_epoch(self) -> Tuple[float, float, float, float, float]: 
        self.model.train()
        train_loss = 0
        epoch_source_label_loss = 0
        epoch_source_domain_loss = 0
        epoch_target_domain_loss = 0
        epoch_domain_loss = 0
        for (X_source_train_batch, y_source_train_batch), (X_target_train_batch, y_target_train_batch) \
        in  zip(self.source_train_loader, self.target_train_loader):

            source_labels_pred, source_domains_pred = self.model(X_source_train_batch, self.grl_lambda)
            _, target_domains_pred = self.model(X_target_train_batch, self.grl_lambda, only_domain=True)
            source_train_batch_labels, source_train_batch_domains = y_source_train_batch[:, 0].long(), y_source_train_batch[:, 1].view(-1, 1)
            target_train_batch_domains = y_target_train_batch[:, 1].view(-1, 1)

            source_label_loss = self.label_loss_function(source_labels_pred, source_train_batch_labels)
            source_domain_loss = self.domain_loss_function(source_domains_pred, source_train_batch_domains)
            target_domain_loss = self.domain_loss_function(target_domains_pred, target_train_batch_domains)
            domain_loss = source_domain_loss + target_domain_loss
            total_loss = source_label_loss + domain_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            epoch_source_label_loss += source_label_loss.item()
            epoch_source_domain_loss += source_domain_loss.item()
            epoch_target_domain_loss += target_domain_loss.item()
            epoch_domain_loss += domain_loss.item()

        mean_source_label_loss = epoch_source_label_loss / len(self.source_train_loader)
        mean_domain_loss = epoch_domain_loss / len(self.source_train_loader)
        mean_source_domain_loss = epoch_source_domain_loss / len(self.source_train_loader)
        mean_target_domain_loss = epoch_target_domain_loss / len(self.source_train_loader)
        return mean_source_label_loss, mean_domain_loss, mean_source_domain_loss, mean_target_domain_loss    

    
    def _evaluate_epoch(self) -> float:
        self.model.eval()
        val_loss = 0
        val_source_label_loss = 0
        val_domain_loss = 0
        with torch.inference_mode():
            for (X_source_val_batch, y_source_val_batch), (X_target_val_batch, y_target_val_batch) \
            in  zip(self.source_val_loader, self.target_val_loader):

                source_labels_pred, source_domains_pred = self.model(X_source_val_batch, self.grl_lambda)
                _, target_domains_pred = self.model(X_target_val_batch, self.grl_lambda, only_domain=True)
                source_val_batch_labels, source_val_batch_domains = y_source_val_batch[:, 0].long(), y_source_val_batch[:, 1].view(-1, 1)
                target_val_batch_domains = y_target_val_batch[:, 1].view(-1, 1)
                
                source_label_loss = self.label_loss_function(source_labels_pred, source_val_batch_labels)
                source_domain_loss = self.domain_loss_function(source_domains_pred, source_val_batch_domains)
                target_domain_loss = self.domain_loss_function(target_domains_pred, target_val_batch_domains)
                domain_loss = source_domain_loss + target_domain_loss
                val_loss += source_label_loss.item() - domain_loss.item()
                val_source_label_loss += source_label_loss.item()
                val_domain_loss -= domain_loss.item()
                
        mean_val_loss = val_loss / len(self.source_val_loader)
        mean_val_source_label_loss = val_source_label_loss / len(self.source_val_loader)
        mean_val_domain_loss = val_domain_loss / len(self.source_val_loader)
        return mean_val_loss, mean_val_source_label_loss, mean_val_domain_loss


    def train_model(self) -> None:
        self.log(self.model._get_name())
        for self.epoch in range(self.epochs):
            self.update_grl_lambda()
            source_label_loss, domain_loss, source_domain_loss, target_domain_loss = self._train_epoch()
            if self.epoch % 5 == 0:
                val_loss, val_source_label_loss, val_domain_loss = self._evaluate_epoch()
                self.log(f"Val Loss: {val_loss:.3f}", f"Val Label Loss: {val_source_label_loss:.3f}", f"Val Domain Loss: {val_domain_loss:.3f}", sep=" ║ ")
            self.early_stopping.check_early_stop(self.model, val_loss, self.epoch)
            if self.early_stopping.should_stop_training():
                self.log(f"Early stopping at epoch {self.epoch}")
                break

        self.model.load_state_dict(self.early_stopping.get_best_model_state_dict())
