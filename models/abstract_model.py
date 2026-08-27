from abc import ABC, abstractmethod

from torch import nn
import torch

class AbstractModel(nn.Module, ABC):
    @abstractmethod
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        pass

    def predict_probs(self, X: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.predict(X), dim=1)
