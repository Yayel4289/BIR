import torch 
from torch import nn

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32):
        super().__init__()
        self.f1 = nn.Linear(input_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = torch.relu(self.f1(X))
        y = torch.sigmoid(self.f2(X))
        return y

