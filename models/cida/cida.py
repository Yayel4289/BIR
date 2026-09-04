from models.dann.dann import DANN

class CIDA(DANN):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32) -> None:
        super().__init__(input_dim, output_dim, hidden_dim)
