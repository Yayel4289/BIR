from models.dann.dann import DANN

class CIDA(DANN):
    def __init__(self, input_dim: int = 2, output_dim: int = 2, hidden_dim: int = 32, convolutional: bool = False, _domain_output_dim: int = 1) -> None:
        super().__init__(input_dim, output_dim, hidden_dim, convolutional, _domain_output_dim)
