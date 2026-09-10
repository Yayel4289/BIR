import torch

from copy import deepcopy
from models.abstract_model import AbstractModel

class ModelSaver:
    def __init__(self, early_stopping_enabled: bool, patience: int=100, delta: float=1e-4):
        self.early_stopping_enabled = early_stopping_enabled
        self.patience = patience
        self.delta = delta
        self.best_loss = torch.inf
        self.best_model_state_dict = None
        self.no_improvement_count = 0
        self.stop_training = False
    
    def update(self, model: AbstractModel, val_loss: float, epoch: int):
        if val_loss < self.best_loss - self.delta or self.best_model_state_dict is None:
            self.best_loss = val_loss
            self.best_model_state_dict = deepcopy(model.state_dict())
            self.no_improvement_count = 0
        elif self.early_stopping_enabled:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True

    def should_stop_training(self):
        return self.stop_training

    def get_best_model_state_dict(self):
        return self.best_model_state_dict
