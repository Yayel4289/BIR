from typing import Tuple
import torch
from torch import nn
from models.dann.gradient_reversal_layer import GradientReversalLayer

# Shallow NN from DANN paper
class DANN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int=32, grl_lambda: float=1) -> None:
        # output_dim = L
        super().__init__()
        self.feature_extractor = nn.Linear(input_dim, hidden_dim) 
        self.label_predictor = nn.Linear(hidden_dim, output_dim)
        self.grl_lambda = grl_lambda
        self.domain_classifier = nn.Linear(hidden_dim, 1) # discriminator, predicts source domain (0) or target domain (1)

    def forward(self, X: torch.Tensor, only_domain=False) -> Tuple[torch.Tensor, torch.Tensor]:
        XF = torch.sigmoid(self.feature_extractor(X))
        label = torch.softmax(self.label_predictor(XF), dim=0) if not only_domain else None
        XF_grl = GradientReversalLayer.apply(XF, self.grl_lambda)
        domain = torch.sigmoid(self.domain_classifier(XF_grl))
        return label, domain

