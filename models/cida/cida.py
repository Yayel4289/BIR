import torch
from models.dann.dann import DANN

class CIDA(DANN):
    def __init__(self, input_dim: int = 2, output_dim: int = 2, hidden_dim: int = 32, convolutional: bool = False, _domain_output_dim: int = 1, device: torch.device = None) -> None:
        super().__init__(input_dim, output_dim, hidden_dim, convolutional, _domain_output_dim, device)

    def forward(self, X: torch.Tensor, grl_lambda: float, only_domain=False) -> Tuple[torch.Tensor, torch.Tensor]:
        label, domain = super().forward(X, grl_lambda, only_domain)
        return label, torch.sigmoid(domain)
