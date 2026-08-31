import torch
from torch import nn
from torch.utils.data import DataLoader
from models.dann.dann import DANN
import optuna
from src.training.trainers.dann_trainer import DANN_Trainer

def select_gamma(
        source_train_loader: DataLoader,
        source_val_loader: DataLoader,
        target_train_loader: DataLoader,
        target_val_loader: DataLoader,
        n_trials: int=30, 
        min_gamma: int=1, 
        max_gamma: int=100, 
        verbose: bool=True
        ) -> float:
    def objective(trial: optuna.Trial) -> float:
        gamma = trial.suggest_float("gamma", min_gamma, max_gamma)

        dann = DANN(input_dim=2, output_dim=2)
        trainer = DANN_Trainer(
            dann,
            source_train_loader,
            source_val_loader,
            target_train_loader,
            target_val_loader,
            nn.CrossEntropyLoss(),
            nn.BCEWithLogitsLoss(),
            gamma=gamma,
        )
        trainer.train_model()

        val_loss = trainer.get_best_val_loss()

        trial.set_user_attr("model_state", dann.state_dict())
        return val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_gamma = study.best_params["gamma"]
    if verbose:
        print(f"Best gamma : {best_gamma} (val loss={study.best_value:.3f})")
    return best_gamma
