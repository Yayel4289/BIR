from models.cida.cida import CIDA
import torch
from torch import nn


class PCIDA(CIDA):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32) -> None:
        super().__init__(input_dim, output_dim, hidden_dim, domain_output_dim=2)

    def predict_domain_prob(self, X: torch.Tensor) -> torch.Tensor:
        XF = self.feature_extractor(X)
        domain_pred = self.domain_classifier(XF)
        domain_mean, domain_var = domain_pred[:, 0], domain_pred[:, 1]
        domain_var = torch.relu(domain_var) + 1e-6
        # 0 = source
        domain_prob = (1 / torch.sqrt(2 * torch.pi * domain_var)) * torch.exp(- ((0 - domain_mean)**2) / (2*domain_var))
        return domain_prob
