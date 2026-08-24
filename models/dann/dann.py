from typing import Tuple
import torch
from torch import nn
from gradient_reversal_layer import GradientReversalLayer

# Shallow NN from DANN paper
class DANN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int=32, lambda_hyperparam: float=1) -> None:
        # output_dim = L
        super().__init__()
        self.feature_extractor = nn.Linear(input_dim, hidden_dim) 
        self.label_predictor = nn.Linear(hidden_dim, output_dim)
        self.grl = GradientReversalLayer(lambda_hyperparam)
        self.domain_classifier = nn.Linear(hidden_dim, 1) # discriminator, predicts source domain (0) or target domain (1)

    def forward(self, X: torch.Tensor, lambda_hyperparam: float=1) -> Tuple[torch.Tensor, float]:
        XF = torch.sigmoid(self.feature_extractor(X))
        y = torch.softmax(self.label_predictor(XF), dim=0)
        XF_grl = self.grl(XF, lambda_hyperparam)
        domain = torch.sigmoid(self.domain_classifier(XF_grl)).item()
        return y, domain

