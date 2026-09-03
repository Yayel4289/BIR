from src.training.lambda_config.abstract_lambda_config import AbstractLambdaConfig

class FixedLambdaConfig(AbstractLambdaConfig):
    def __init__(self, grl_lambda: float) -> None:
        super().__init__(progressive=False)
        self.grl_lambda = grl_lambda
