from typing import Tuple
import torch
from torch import nn
from models.abstract_model import AbstractModel
from models.dann.gradient_reversal_layer import GradientReversalLayer

class DANN(AbstractModel):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int=32, grl_lambda: float=0.1) -> None:
        # output_dim = L
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ) 
        self.label_predictor = nn.Linear(hidden_dim, output_dim)
        self.grl_lambda = grl_lambda
        self.domain_classifier = nn.Linear(hidden_dim, 1) # discriminator, predicts source domain (0) or target domain (1)

    def forward(self, X: torch.Tensor, only_domain=False) -> Tuple[torch.Tensor, torch.Tensor]:
        XF = self.feature_extractor(X)
        label = self.label_predictor(XF) if not only_domain else None
        XF_grl = GradientReversalLayer.apply(XF, self.grl_lambda)
        domain = self.domain_classifier(XF_grl)
        return label, domain

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        XF = self.feature_extractor(X)
        return torch.softmax(self.label_predictor(XF), dim=1)
