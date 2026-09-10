from abc import ABC, abstractmethod

from torch import nn
import torch

class AbstractModel(nn.Module, ABC):
    def __init__(self, input_dim: int = 2, output_dim: int = 2, hidden_dim: int = 32, convolutional: bool = False, device: torch.device = None) -> None:
        super().__init__()
        if device is not None:
            self.to(device)
        self.feature_extractor = self._define_linear_feature_extractor(input_dim, hidden_dim) if not convolutional else \
                                 self._define_convolutional_feature_extractor(input_dim, hidden_dim)

        self.label_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def _define_linear_feature_extractor(self, input_dim: int, hidden_dim: int):
        return nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
        ) 

    def _define_convolutional_feature_extractor(self, input_dim: int, hidden_dim: int):
        return nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # W,H /= 2

            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), #W,H /= 2

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    @abstractmethod
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        pass

    def predict_probs(self, X: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.predict(X), dim=1)
