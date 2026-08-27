from typing import Callable
import torch 
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from models.abstract_model import AbstractModel
from src.training.early_stopping import EarlyStopping

class BaselineTrainer:
    def __init__(
            self, 
            model: AbstractModel, 
            source_train_loader: DataLoader,
            source_val_loader: DataLoader,
            label_loss_function: Callable[torch.Tensor, torch.Tensor], 
            epochs:int=1000, 
            lr:float=0.01,
            min_epoch_thresh: int=0
            ) -> None:
        self.model = model 
        self.source_train_loader = source_train_loader
        self.source_val_loader = source_val_loader
        self.label_loss_function = label_loss_function
        self.epochs = epochs
        self.epoch = 0
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=lr)
        self.early_stopping = EarlyStopping(min_epoch_thresh=min_epoch_thresh)

    def _train_epoch(self) -> float: 
        self.model.train()
        train_loss = 0
        for X_train_batch, y_train_batch in self.source_train_loader:
            y_pred, _ = self.model(X_train_batch)
            y_train_batch_labels = y_train_batch[:, 0].long()
            label_loss = self.label_loss_function(y_pred, y_train_batch_labels)

            self.optimizer.zero_grad()
            label_loss.backward()
            self.optimizer.step()
            train_loss += label_loss.item()

        mean_train_loss = train_loss / len(self.source_train_loader)
        return mean_train_loss

    def _evaluate_epoch(self) -> float:
        self.model.eval()
        source_val_loss = 0
        with torch.inference_mode():
            for X_source_val_batch, y_source_val_batch in self.source_val_loader:
                y_source_pred = self.model.predict(X_source_val_batch)
                y_source_val_batch_labels = y_source_val_batch[:, 0].long()
                source_label_loss = self.label_loss_function(y_source_pred, y_source_val_batch_labels)
                source_val_loss += source_label_loss.item()
        mean_source_val_loss = source_val_loss / len(self.source_val_loader)
        return mean_source_val_loss

    def train_model(self) -> None:
        print(self.model._get_name())
        for self.epoch in range(self.epochs):
            train_loss = self._train_epoch()
            val_loss = self._evaluate_epoch()
            if self.epoch % 5 == 0:
                print(f"Train Loss: {train_loss:.3f}", f"Val Loss: {val_loss:.3f}", sep=" ║ ")
            self.early_stopping.check_early_stop(self.model, val_loss, self.epoch)
            if self.early_stopping.should_stop_training():
                print(f"Early stopping at epoch {self.epoch}")
                break

        self.model.load_state_dict(self.early_stopping.get_best_model_state_dict())
