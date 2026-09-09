from typing import Tuple
import torch
from torch import nn
from models.abstract_model import AbstractModel
from models.dann.gradient_reversal_layer import GradientReversalLayer

class DANN(AbstractModel):
    def __init__(self, input_dim: int = 2, output_dim: int = 2, hidden_dim: int=32, convolutional: bool = False, _domain_output_dim: int = 1, device: torch.device = None) -> None:
        # output_dim = L
        super().__init__(input_dim, output_dim, hidden_dim, convolutional, device)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, _domain_output_dim)
        )

    def forward(self, X: torch.Tensor, grl_lambda: float, only_domain=False) -> Tuple[torch.Tensor, torch.Tensor]:
        XF = self.feature_extractor(X)
        label = self.label_predictor(XF) if not only_domain else None
        XF_grl = GradientReversalLayer.apply(XF, grl_lambda)
        domain = self.domain_classifier(XF_grl)
        return label, domain

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        XF = self.feature_extractor(X)
        return self.label_predictor(XF)

    def predict_domain_prob(self, X: torch.Tensor) -> torch.Tensor:
        XF = self.feature_extractor(X)
        domain_prob = torch.sigmoid(self.domain_classifier(XF))
        return domain_prob
