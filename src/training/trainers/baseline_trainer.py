from typing import Callable, List
import torch 
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from models.abstract_model import AbstractModel
from src.training.model_saver import ModelSaver
from src.visualization.logger import Logger
from src.utils.utils import to_device

class BaselineTrainer:
    def __init__(
            self, 
            model: AbstractModel, 
            source_train_loader: DataLoader,
            source_val_loader: DataLoader,
            label_loss_function: Callable[torch.Tensor, torch.Tensor], 
            epochs:int=1000, 
            lr:float=0.01,
            verbose: bool = True,
            device: torch.device = None
            ) -> None:
        self.model = model 
        self.source_train_loader = source_train_loader
        self.source_val_loader = source_val_loader
        self.label_loss_function = label_loss_function
        self.epochs = epochs
        self.logger = Logger(verbose)
        self.device = device
        self.epoch = 0
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=lr)
        self.model_saver = ModelSaver(True)

    def to_device(self, *tensors: torch.Tensor):
        return to_device(self.device, *tensors)

    def get_best_val_loss(self) -> float:
        return self.model_saver.best_loss

    def _train_epoch(self) -> float: 
        self.model.train()
        train_loss = 0
        for i in range(len(self.source_train_loader)):
        #for X_train_batch, y_train_batch in self.source_train_loader:
            X_train_batch, y_train_batch = next(iter(self.source_train_loader))
            X_train_batch, y_train_batch = self.to_device(X_train_batch, y_train_batch)
            y_pred, _ = self.model(X_train_batch)
            y_train_batch_labels = y_train_batch[:, 0].long()
            label_loss = self.label_loss_function(y_pred, y_train_batch_labels)

            self.optimizer.zero_grad()
            label_loss.backward()
            self.optimizer.step()
            train_loss += label_loss.item()

        mean_train_loss = train_loss / (i+1)
        return mean_train_loss

    def _evaluate_epoch(self) -> float:
        self.model.eval()
        source_val_loss = 0
        with torch.inference_mode():
            for i in range(len(self.source_val_loader)):
            #for X_source_val_batch, y_source_val_batch in self.source_val_loader:
                X_source_val_batch, y_source_val_batch = next(iter(self.source_val_loader))
                X_source_val_batch, y_source_val_batch = self.to_device(X_source_val_batch, y_source_val_batch)
                y_source_pred = self.model.predict(X_source_val_batch)
                y_source_val_batch_labels = y_source_val_batch[:, 0].long()
                source_label_loss = self.label_loss_function(y_source_pred, y_source_val_batch_labels)
                source_val_loss += source_label_loss.item()
        mean_source_val_loss = source_val_loss / (i+1)
        return mean_source_val_loss

    def train_model(self) -> None:
        self.logger(self.model._get_name())
        for self.epoch in range(self.epochs):
            train_loss = self._train_epoch()
            val_loss = self._evaluate_epoch()
            print(f"Epoch {self.epoch+1}/{self.epochs} ║ Train Loss: {train_loss:.3f} ║ Val Loss: {val_loss:.3f}", flush=True)
            self.model_saver.update(self.model, val_loss, self.epoch)
            if self.model_saver.should_stop_training():
                self.logger(f"Early stopping at epoch {self.epoch}")
                break

        self.model.load_state_dict(self.model_saver.get_best_model_state_dict())
