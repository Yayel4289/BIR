from typing import List
import torch

from copy import deepcopy
from models.abstract_model import AbstractModel

class EarlyStopping:
    def __init__(self, loss_size: int=1, patience: int=100, delta: float=1e-4):
        self.loss_size = loss_size
        self.patience = patience
        self.delta = delta
        self.best_losses = [torch.inf] * self.loss_size
        self.best_model_state_dict = None
        self.no_improvement_count = 0
        self.stop_training = False

    def _update_losses(self, val_losses: List[float]) -> bool: # True if loss update, else False
        update = False
        for i, loss in enumerate(val_losses):
            if loss < self.best_losses[i] - self.delta:
                self.best_losses[i] = loss
                update = True
        return update
    
    def check_early_stop(self, model: AbstractModel, val_losses: List[float], epoch: int):
        if len(val_losses) != self.loss_size:
            raise ValueError(f"Losses must be of size {self.loss_size}")
        updated = self._update_losses(val_losses)
        if updated:
            self.best_model_state_dict = deepcopy(model.state_dict())
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = False

    def should_stop_training(self):
        return self.stop_training

    def get_best_model_state_dict(self):
        return self.best_model_state_dict
