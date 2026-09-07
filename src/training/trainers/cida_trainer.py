from torch import nn

from src.training.trainers.dann_trainer import DANN_Trainer


class CIDA_Trainer(DANN_Trainer):
    def __init__(
            self, 
            model: CIDA, 
            source_train_loader: DataLoader, 
            source_val_loader: DataLoader, 
            target_train_loader: DataLoader, 
            target_val_loader: DataLoader, 
            lambda_config: AbstractLambdaConfig = ..., 
            epochs: int = 1000, 
            lr: float = 0.01, 
            verbose: bool = True
    ) -> None:
        super().__init__(
            model, 
            source_train_loader, source_val_loader, target_train_loader, target_val_loader, 
            lambda_config, 
            epochs, 
            lr, 
            verbose
        )

    def _define_domain_loss_func(self) -> Callable[torch.Tensor, torch.Tensor]:
        return nn.MSELoss()
