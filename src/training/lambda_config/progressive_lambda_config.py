import numpy as np
from src.training.lambda_config.abstract_lambda_config import AbstractLambdaConfig


class ProgressiveLambdaConfig(AbstractLambdaConfig):
    def __init__(self, max_grl_lambda: int, gamma: int) -> None:
        super().__init__(progressive=True)
        self.max_grl_lambda = max_grl_lambda
        self.gamma = gamma

    def update_grl_lambda(self, epoch: int, epochs: int) -> float:
        p = epoch / epochs
        self.grl_lambda = self.max_grl_lambda * (2 / (1 + np.exp(-self.gamma * p)) - 1)

