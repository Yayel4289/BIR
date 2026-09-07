from typing import Tuple, Type
import torch
from torch import nn
from torch.utils.data import DataLoader
from models.dann.dann import DANN
import optuna
from src.training.trainers.dann_trainer import DANN_Trainer
from src.visualization.logger import Logger
from src.training.lambda_config.progressive_lambda_config import ProgressiveLambdaConfig
from src.training.lambda_config.fixed_lambda_config import FixedLambdaConfig

def select_model_progressive_hyperparams(
        seed: int,
        ModelClass: Type[DANN],
        TrainerClass: Type[DANN_Trainer],
        source_train_loader: DataLoader,
        source_val_loader: DataLoader,
        target_train_loader: DataLoader,
        target_val_loader: DataLoader,
        n_trials: int=30, 
        min_gamma: int=1, 
        max_gamma: int=50, 
        min_max_grl_lambda: int=1,
        max_max_grl_lambda: int=15,
        verbose: bool=False
        ) -> Tuple[float, float]:
    def objective(trial: optuna.Trial) -> float:
        gamma = trial.suggest_int("gamma", min_gamma, max_gamma)
        max_grl_lambda = trial.suggest_int("max_grl_lambda", min_max_grl_lambda, max_max_grl_lambda)

        model = ModelClass(input_dim=2, output_dim=2)
        trainer = TrainerClass(
            model,
            source_train_loader,
            source_val_loader,
            target_train_loader,
            target_val_loader,
            lambda_config=ProgressiveLambdaConfig(max_grl_lambda, gamma),
            verbose=verbose
        )
        trainer.train_model()
        val_loss = trainer.get_best_val_loss()
        return val_loss

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.FATAL)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_gamma = study.best_params["gamma"]
    best_max_grl_lambda = study.best_params["max_grl_lambda"]
    Logger.static_log(verbose, f"Best gamma : {best_gamma}, best max_grl_lambda : {best_max_grl_lambda} (val loss={study.best_value:.3f})")
    return best_max_grl_lambda, best_gamma


def select_model_fixed_grl_lambda(
        seed: int,
        ModelClass: Type[DANN],
        TrainerClass: Type[DANN],
        source_train_loader: DataLoader,
        source_val_loader: DataLoader,
        target_train_loader: DataLoader,
        target_val_loader: DataLoader,
        n_trials: int=30, 
        min_grl_lambda: float = 0.1,
        max_grl_lambda: float = 10,
        verbose: bool=False
        ) -> float:
    def objective(trial: optuna.Trial) -> float:
        grl_lambda = trial.suggest_float("grl_lambda", min_grl_lambda, max_grl_lambda)

        model = ModelClass(input_dim=2, output_dim=2)
        trainer = TrainerClass(
            model,
            source_train_loader,
            source_val_loader,
            target_train_loader,
            target_val_loader,
            lambda_config=FixedLambdaConfig(grl_lambda),
            verbose=verbose
        )
        trainer.train_model()
        val_loss = trainer.get_best_val_loss()
        return val_loss

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.FATAL)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_grl_lambda = study.best_params["grl_lambda"]
    Logger.static_log(verbose, f"Best grl_lambda : {best_grl_lambda} (val loss={study.best_value:.3f})")
    return best_grl_lambda

