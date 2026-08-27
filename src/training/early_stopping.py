import torch

class EarlyStopping:
    def __init__(self, patience=100, delta=1e-5):
        self.patience = patience
        self.delta = delta
        self.best_loss = torch.inf
        self.no_improvement_count = 0
        self.stop_training = False
    
    def check_early_stop(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True

    def should_stop_training(self):
        return self.stop_training
