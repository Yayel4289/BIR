from models.cida.cida import CIDA
import torch
from torch import nn


class PCIDA(CIDA):
    def __init__(self, input_dim: int = 2, output_dim: int = 2, hidden_dim: int = 32, convolutional: bool = False, _domain_output_dim: int = 2, device: torch.device = None) -> None:
        super().__init__(input_dim, output_dim, hidden_dim, convolutional, _domain_output_dim, device)

    def predict_domain_prob(self, X: torch.Tensor) -> torch.Tensor:
        XF = self.feature_extractor(X)
        domain_pred = self.domain_classifier(XF)
        domain_mean, domain_logvar = domain_pred[:, 0], domain_pred[:, 1]
        domain_logvar = torch.clamp(domain_logvar, min=1e-6)
        # 0 = source
        domain_prob = (1 / torch.sqrt(2 * torch.pi * torch.exp(domain_logvar))) * torch.exp(- ((0 - domain_mean)**2) / (2*torch.exp(domain_logvar)))
        return domain_prob
