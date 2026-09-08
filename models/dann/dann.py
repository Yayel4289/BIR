from typing import Tuple
import torch
from torch import nn
from models.abstract_model import AbstractModel
from models.dann.gradient_reversal_layer import GradientReversalLayer

class DANN(AbstractModel):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int=32, domain_output_dim: int = 1, convolutional: bool = False) -> None:
        # output_dim = L
        super().__init__()
        half_hidden_dim = hidden_dim // 2
        if not convolutional:
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, half_hidden_dim),
                nn.ReLU(),
                nn.Linear(half_hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) 
        else:
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, half_hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2), # W,H /= 2

                nn.Conv2d(half_hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2), #W,H /= 2

                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),

                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
        self.label_predictor = nn.Sequential(
            nn.Linear(hidden_dim, half_hidden_dim),
            nn.ReLU(),
            nn.Linear(half_hidden_dim, output_dim)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, half_hidden_dim),
            nn.ReLU(),
            nn.Linear(half_hidden_dim, domain_output_dim)
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
