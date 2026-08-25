from typing import Callable
import torch 
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

class BaselineTrainer:
    def __init__(
            self, 
            model: nn.Module, 
            source_train_loader: DataLoader,
            source_val_loader: DataLoader,
            label_loss_function: Callable[torch.Tensor, torch.Tensor], 
            epochs:int=100, 
            lr:float=0.01
    ) -> None:
        self.model = model 
        self.source_train_loader = source_train_loader
        self.source_val_loader = source_val_loader
        self.label_loss_function = label_loss_function
        self.epochs = epochs
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=lr)

    def _train_epoch(self) -> float: 
        self.model.train()
        train_loss = 0
        for X_train_batch, y_train_batch in self.source_train_loader:
            y_pred = self.model(X_train_batch)
            y_train_batch_labels = y_train_batch[:, 0].view(-1, 1)
            label_loss = self.label_loss_function(y_pred, y_train_batch_labels)

            self.optimizer.zero_grad()
            label_loss.backward()
            self.optimizer.step()
            
            train_loss += label_loss.item()

        mean_train_loss = train_loss / len(self.source_train_loader)
        return mean_train_loss

    def _evaluate_epoch(self) -> float:
        self.model.eval()
        val_loss = 0
        with torch.inference_mode():
            for X_val_batch, y_val_batch in self.source_val_loader:
                y_pred = self.model(X_val_batch)
                y_val_batch_labels = y_val_batch[:, 0].view(-1, 1)
                label_loss = self.label_loss_function(y_pred, y_val_batch_labels)
                val_loss += label_loss.item()
        mean_val_loss = val_loss / len(self.source_val_loader)
        return mean_val_loss

    def train_model(self) -> None:
        print(self.model._get_name())
        for epoch in range(self.epochs):
            train_loss = self._train_epoch()
            if epoch % 5 == 0:
                val_loss = self._evaluate_epoch()
                print(f"Train Loss: {train_loss}", f"Test Loss: {val_loss}", sep="\t")
