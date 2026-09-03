from abc import ABC, abstractmethod

class AbstractLambdaConfig(ABC):
    def __init__(self, progressive: bool) -> None:
        super().__init__()
        self.progressive = progressive
        self.grl_lambda: float

    def is_progressive(self) -> bool:
        return self.progressive

    def get_grl_lambda(self) -> float:
        return self.grl_lambda

    def update_grl_lambda(self, **kwargs) -> None: # can be overridden by progressives 
        pass

