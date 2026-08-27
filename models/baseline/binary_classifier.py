from typing import Tuple
import torch 
from torch import nn

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        #self.f1 = nn.Linear(input_dim, hidden_dim)
        #self.f2 = nn.Linear(hidden_dim, 2)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.NoneType]:
        return self.fc(X), None
        #X = torch.relu(self.f1(X))

        #y = self.f2(X) # softmax applied with nn.CrossEntropyLoss()
        #return y, None # label, domain -> no concern for domain
