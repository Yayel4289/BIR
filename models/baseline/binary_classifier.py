from typing import Tuple
import torch 
from torch import nn

from models.abstract_model import AbstractModel

class BinaryClassifier(AbstractModel):
    def __init__(self, input_dim=2, hidden_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        #self.f1 = nn.Linear(input_dim, hidden_dim)
        #self.f2 = nn.Linear(hidden_dim, 2)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.NoneType]:
        return self.fc(X), None

    def predict(self, X:torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.fc(X), dim=1)
