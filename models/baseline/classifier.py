from typing import Tuple
import torch 
from torch import nn

from models.abstract_model import AbstractModel

class Classifier(AbstractModel):
    def __init__(self, input_dim=2, output_dim=2, hidden_dim=32, convolutional: bool = False, device: torch.device = None):
        super().__init__(input_dim, output_dim, hidden_dim, convolutional, device)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.NoneType]:
        XF = self.feature_extractor(X)
        return self.label_predictor(XF), None

    def predict(self, X:torch.Tensor) -> torch.Tensor:
        XF = self.feature_extractor(X)
        return self.label_predictor(XF)
