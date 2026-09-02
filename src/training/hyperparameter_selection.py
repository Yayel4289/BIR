import torch
from torch import nn
from torch.utils.data import DataLoader
from models.dann.dann import DANN
import optuna
from src.training.trainers.dann_trainer import DANN_Trainer

def select_DANN_hyperparams(
        seed: int,
        source_train_loader: DataLoader,
        source_val_loader: DataLoader,
        target_train_loader: DataLoader,
        target_val_loader: DataLoader,
        n_trials: int=30, 
        min_gamma: int=1, 
        max_gamma: int=100, 
        min_max_lambda: int=1,
        max_max_lambda: int=15,
        verbose: bool=True
        ) -> float:
    def objective(trial: optuna.Trial) -> float:
        gamma = trial.suggest_int("gamma", min_gamma, max_gamma)
        max_lambda = trial.suggest_int("max_lambda", min_max_lambda, max_max_lambda)

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
            max_lambda=max_lambda
        )
        trainer.train_model()

        val_loss = trainer.get_best_val_loss()

        return val_loss

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_gamma = study.best_params["gamma"]
    best_max_lambda = study.best_params["max_lambda"]
    if verbose:
        print(f"Best gamma : {best_gamma}, best max_lambda : {best_max_lambda} (val loss={study.best_value:.3f})")
    return best_gamma, best_max_lambda
