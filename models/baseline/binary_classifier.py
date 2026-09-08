from typing import Tuple
import torch 
from torch import nn

from models.abstract_model import AbstractModel

class BinaryClassifier(AbstractModel):
    def __init__(self, input_dim=2, hidden_dim=32, convolutional: bool = False):
        super().__init__()
        if not convolutional:
            self.fc = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)
            )
        else:
            self.fc = nn.Sequential(
                nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2), # W,H /= 2

                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2), #W,H /= 2

                nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1),
                nn.ReLU(),

                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.NoneType]:
        return self.fc(X), None

    def predict(self, X:torch.Tensor) -> torch.Tensor:
        return self.fc(X)
