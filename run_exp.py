"""
Usage:
    run_exp -h
    run_exp --moons --plot-only [--continuous]
    run_exp (--moons | --skin-cancer) --model <model_name> [--mode <mode_name> [--hyperparam-selection]]
    run_exp (--moons | --skin-cancer) --comparison [--model <model_name> [--mode <mode_name>]]

Options:
    -h                                          Show this help.

    --moons                                     Run the intertwinning mooons toy problem.

    --skin-cancer                               Run the skin cancer classification problem.

    --plot-only                                 Show only intertwinning moons plot (continuous domains or not, see --continuous).

    --continuous                                Decides if the domains are continuous or not.

    --model <model_name>                        Run <model_name> model (see --mode for models modes).
                                                <model_name> can either be 'baseline', 'dann', 'cida' or 'pcida'.

    --mode <mode_name>                          Specify the mode of the model
                                                <mode_name> can either be 'fixed' or 'progressive'. 
                                                This refers to the evolution of grl_lambda through the training of DANN, CIDA or PCIDA.
                                                Only available for DANN, CIDA and PCIDA.

    --hyperparam-selection                      Perform an hyperparameter selection depending on the <mode>.
                                                If <mode_name> is 'fixed', the hyperparameter will be grl_lambda.
                                                If <mode_name> is 'progressive', the hyperparameters will be max_grl_lambda and gamma, 
                                                deciding the evolution curve of grl_lambda.
                                                Only available for DANN, CIDA or PCIDA.

    --comparison                                Compare models based on H-Divergence and F1 Score
                                                (Both hyperparameter selections for domain adaptation models)
"""

from typing import Callable
from docopt import docopt
from colorama import Fore, Style
from tqdm import tqdm
import sys
import torch 
from torch import nn
import numpy as np 
import pandas as pd

from exps.skinCancer.data_loader import get_skin_cancer_data_loader_set
from src.data.data_loader_set import DataLoaderSet
from exps.points_classification.data.data_generation import generate_fused_domains, generate_intertwinning_moons, generate_split_domains, get_X_from_source_target, get_moons_data_loader_set, get_source_target_loaders_from_points, get_train_val_loaders_from_source, get_train_val_loaders_from_source_target
from exps.points_classification.utils.point import Point
from models.abstract_model import AbstractModel
from models.cida.cida import CIDA
from models.cida.pcida import PCIDA
from src.metrics.metrics import compute_model_f1_score_from_loader, compute_model_h_divergence, compute_model_source_f1_score, compute_model_target_f1_score, get_model_metrics, get_table_string, has_model_h_divergence, make_metrics_dict, udpate_model_metrics_mean_std, update_model_metrics_lists
from models.baseline.classifier import Classifier
from models.dann.dann import DANN
import src.visualization.plots as plots
from src.training.trainers.baseline_trainer import BaselineTrainer
from src.training.trainers.dann_trainer import DANN_Trainer
from src.training.hyperparameter_selection import select_model_progressive_hyperparams, select_model_fixed_grl_lambda
from src.training.lambda_config.progressive_lambda_config import ProgressiveLambdaConfig
from src.training.lambda_config.fixed_lambda_config import FixedLambdaConfig
from src.training.trainers.cida_trainer import CIDA_Trainer
from src.training.trainers.pcida_trainer import PCIDA_Trainer
from typing import Type, Dict, List, Tuple
import sklearn

DEFAULT_SEED = 42
COMPARISON_SEEDS = [DEFAULT_SEED, 89, 0, 4289]

def get_trainer_model_from_model_name(formatted_model_name: str) -> Tuple[Type[DANN], Type[DANN_Trainer]]:
    acronym = formatted_model_name.upper()
    return globals().get(acronym), globals().get(acronym + "_Trainer")

def is_model_continuous(formatted_model_name: str) -> bool:
    cida = "cida"
    return cida in formatted_model_name or cida.upper() in formatted_model_name

def run_baseline(
        data_loader_set: DataLoaderSet, 
        plot: bool = True, 
        verbose: bool = True, 
        device: torch.device = None, 
        loss_kwargs: Dict = {}, 
        training_kwargs: Dict = {},
        model_kwargs: Dict = {}
        ) -> Classifier:
    source_train_loader, source_val_loader, source_test_loader, _, _, target_test_loader = data_loader_set

    baseline = Classifier(device=device, **model_kwargs)
    baseline_trainer = BaselineTrainer(
        baseline, 
        source_train_loader, 
        source_val_loader, 
        nn.CrossEntropyLoss(**loss_kwargs),
        verbose=verbose,
        epochs=training_kwargs["epochs"],
        device=device
    )
    baseline_trainer.train_model()
    if verbose:
        source_f1_score = compute_model_f1_score_from_loader(baseline, source_test_loader, device=device)
        target_f1_score = compute_model_f1_score_from_loader(baseline, target_test_loader, device=device)
        print("Source F1 Score:", source_f1_score)
        print("Target F1 Score:", target_f1_score)
    if plot and data_loader_set.has_test_pts():
        plot_model(data_loader_set.get_test_pts(), baseline)
    return baseline

def run_adversarial_model(
        data_loader_set: DataLoaderSet, 
        formatted_model_name: str, 
        mode: str, 
        hyperparam_selection: bool, 
        plot: bool = True, 
        verbose: bool = True, 
        device: torch.device = None, 
        model_kwargs: Dict = {}, 
        training_kwargs: Dict = {},
        loss_kwargs: Dict = {}
        ) -> DANN:
    source_train_loader, source_val_loader, source_test_loader, target_train_loader, target_val_loader, target_test_loader = data_loader_set
    ModelCass, TrainerClass = get_trainer_model_from_model_name(formatted_model_name)
    
    kwargs = {"epochs": training_kwargs["epochs"]}
    if hyperparam_selection:
        if mode == "fixed":
            grl_lambda = select_model_fixed_grl_lambda(
                DEFAULT_SEED,
                ModelCass, TrainerClass,
                source_train_loader, source_val_loader, target_train_loader, target_val_loader,
                verbose=verbose,
                device=device,
                loss_kwargs=loss_kwargs,
                model_kwargs=model_kwargs,
                **training_kwargs
            )
            kwargs["lambda_config"] = FixedLambdaConfig(grl_lambda)
        elif mode == "progressive":
            max_grl_lambda, gamma = select_model_progressive_hyperparams(
                DEFAULT_SEED, 
                ModelCass, TrainerClass,
                source_train_loader, source_val_loader, target_train_loader, target_val_loader, 
                verbose=verbose,
                device=device,
                loss_kwargs=loss_kwargs,
                model_kwargs=model_kwargs,
                **training_kwargs
            )
            kwargs["lambda_config"] = ProgressiveLambdaConfig(max_grl_lambda, gamma)
    else:
        kwargs["lambda_config"] = ProgressiveLambdaConfig(1, 10) if mode == "progressive" else FixedLambdaConfig(1)

    model = ModelCass(device=device, **model_kwargs)
    trainer = TrainerClass(
        model, 
        source_train_loader, source_val_loader, target_train_loader, target_val_loader,
        verbose=verbose,
        device=device,
        loss_kwargs=loss_kwargs,
        **kwargs
    )
    trainer.train_model()
    
    if verbose:
        source_f1_score = compute_model_f1_score_from_loader(model, source_test_loader, device=device)
        target_f1_score = compute_model_f1_score_from_loader(model, target_test_loader, device=device)
        h_divergence = compute_model_h_divergence(model, source_test_loader, target_test_loader, device=device)
        print("Source F1 Score:", source_f1_score)
        print("Target F1 Score:", target_f1_score)
        print("H-Divergence:", h_divergence)
    if plot and data_loader_set.has_test_pts():
        plot_model(data_loader_set.get_test_pts(), model)
    return model


def make_table_comparison(
        exp: str, 
        device: torch.device, 
        formatted_model_name: str = None, 
        mode_name: str = None, 
        table: bool = True, 
        model_kwargs: Dict = {}, 
        training_kwargs: Dict = {},
        loss_kwargs: Dict = {}
    ) -> None:
    data_loader_set_function: Callable[bool, DataLoaderSet]
    if exp == "moons":
        data_loader_set_function = get_moons_data_loader_set
    elif exp == "skin-cancer":
        data_loader_set_function = get_skin_cancer_data_loader_set

    models_config =  {
        "Baseline": {
            "run_func": run_baseline,
            "metrics": make_metrics_dict(h_divergence=False),
            "kwargs": {
                "verbose": False,
                "plot": False
            }
        },
        "DANN Fixed λ": {
            "run_func": run_adversarial_model,
            "metrics": make_metrics_dict(),
            "args": ["dann", "fixed", True],
            "kwargs": {
                "verbose": False,
                "plot": False
            }
        },
        "DANN Progressive λ": {
            "run_func": run_adversarial_model,
            "metrics": make_metrics_dict(),
            "args": ["dann", "progressive", True],
            "kwargs": {
                "verbose": False,
                "plot": False
            }
        },
        "CIDA Fixed λ": {
            "run_func": run_adversarial_model,
            "metrics": make_metrics_dict(),
            "args": ["cida", "fixed", True],
            "kwargs": {
                "verbose": False,
                "plot": False
            }
        },
        "CIDA Progressive λ": {
            "run_func": run_adversarial_model,
            "metrics": make_metrics_dict(),
            "args": ["cida", "progressive", True],
            "kwargs": {
                "verbose": False,
                "plot": False
            }
        },
        "PCIDA Fixed λ": {
            "run_func": run_adversarial_model,
            "metrics": make_metrics_dict(),
            "args": ["pcida", "fixed", True],
            "kwargs": {
                "verbose": False,
                "plot": False
            }
        },
        "PCIDA Progressive λ": {
            "run_func": run_adversarial_model,
            "metrics": make_metrics_dict(),
            "args": ["pcida", "progressive", True],
            "kwargs": {
                "verbose": False,
                "plot": False
            }
        }
    }

    if formatted_model_name is not None:
        if mode_name is None:
            models_config = {"Baseline": models_config["Baseline"]}
        else:
            key = formatted_model_name.upper() + " " + mode_name.capitalize() + " λ"
            models_config = {key: models_config[key]}

    for i, seed in enumerate(tqdm(COMPARISON_SEEDS)):
        setup_seed(seed)
        discrete_data_loader_set = data_loader_set_function(continuous=False)
        continuous_data_loader_set = data_loader_set_function(continuous=True)
        for model_name in models_config:
            data_loader_set = continuous_data_loader_set if is_model_continuous(model_name) else discrete_data_loader_set
            run_func = models_config[model_name]["run_func"]
            args = models_config[model_name].get("args")
            kwargs = models_config[model_name]["kwargs"]
            model = run_func(
                data_loader_set, 
                device=device, 
                model_kwargs=model_kwargs, 
                training_kwargs=training_kwargs, 
                loss_kwargs=loss_kwargs, 
                **kwargs
            ) if args is None else \
                    run_func(
                data_loader_set, 
                *args, 
                device=device, 
                model_kwargs=model_kwargs, 
                training_kwargs=training_kwargs, 
                loss_kwargs=loss_kwargs, 
                **kwargs
            )
            
            model_metrics = models_config[model_name]["metrics"]
            source_test_loader, target_test_loader = data_loader_set.get_test_loaders()
            update_model_metrics_lists(model_metrics, model, source_test_loader, target_test_loader, device)
            if i == len(COMPARISON_SEEDS) - 1: # last pass
                udpate_model_metrics_mean_std(model_metrics)

    if table:
        print(get_table_string(models_config), flush=True)
    else:
        print(get_model_metrics(models_config), flush=True)

 
def plot_model(points: List[Point], model: nn.Module):
    plots.plot_pts(points, show_target_label=True)
    plots.plot_decision_boundary(points, model)
    plots.show()

def plot_pts_only(points: List[Point]):
    plots.plot_pts(points)
    plots.show()

def print_help_and_exit(optional_msg=""):
    if optional_msg:
        print(optional_msg)
    print(__doc__)
    sys.exit(1)

def setup_seed(seed: int = DEFAULT_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

def validate_model_params(formatted_model_name: str, mode_name: str, hyperparam_selection: bool) -> bool: # True if model mode, else False
    if formatted_model_name is None:
        return False
    if formatted_model_name not in ["baseline", "dann", "cida", "pcida"]: 
        print_help_and_exit(Fore.RED + f"Wrong model name: '{formatted_model_name}'" + Style.RESET_ALL)
    elif mode_name is not None and mode_name not in ["fixed", "progressive"]:
        print_help_and_exit(Fore.RED + f"Wrong mode: '{mode_name}'" + Style.RESET_ALL)
    elif formatted_model_name == "baseline" and (hyperparam_selection or mode_name is not None):
        print_help_and_exit(Fore.RED + f"Hyperparameter selection or mode is not available for baseline" + Style.RESET_ALL)
    return True


def main(args) -> None:
    if args["-h"]:
        print_help_and_exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    formatted_model_name = args["--model"].lower() if args["--model"] is not None else None
    mode_name = args["--mode"].lower() if args["--mode"] is not None else None
    hyperparam_selection = args["--hyperparam-selection"]
    model_mode = validate_model_params(formatted_model_name, mode_name, hyperparam_selection)
    exp = "moons" if args["--moons"] else "skin-cancer"

    setup_seed()

    if args["--moons"] and args["--plot-only"]:
        points = generate_intertwinning_moons(continuous=args["--continuous"])
        plot_pts_only(points)
    elif model_mode:
        model_kwargs = {}
        training_kwargs = {"epochs": 1000, "n_trials": 15}
        loss_kwargs = {}
        if exp == "skin-cancer":
            model_kwargs = {"input_dim": 3, "convolutional": True}
            training_kwargs = {"epochs": 500, "n_trials": 15}
            loss_kwargs = {"weight": torch.Tensor([1, 6])} # benign / cancer inverse proportions

        if args["--comparison"]:
            make_table_comparison(
                exp, 
                device, 
                formatted_model_name=formatted_model_name, 
                mode_name=mode_name, 
                model_kwargs=model_kwargs, 
                training_kwargs=training_kwargs,
                loss_kwargs=loss_kwargs, 
                table=False)
            sys.exit(0)
        continuous = is_model_continuous(formatted_model_name)
        data_loader_set: DataLoaderSet
        if args["--moons"]:
            data_loader_set = get_moons_data_loader_set(continuous=continuous)
        elif args["--skin-cancer"]:
            data_loader_set = get_skin_cancer_data_loader_set(continuous=continuous)

        if formatted_model_name == "baseline":
            run_baseline(data_loader_set, device=device, model_kwargs=model_kwargs, training_kwargs=training_kwargs, loss_kwargs=loss_kwargs)
        else:
            run_adversarial_model(
                data_loader_set, 
                formatted_model_name, 
                mode_name, 
                hyperparam_selection, 
                device=device, 
                model_kwargs=model_kwargs, 
                training_kwargs=training_kwargs,
                loss_kwargs=loss_kwargs
            )
    elif args["--comparison"]:
        make_table_comparison(exp, device)
        sys.exit(0)
    else:
        print_help_and_exit()

if __name__ == "__main__":
    args = docopt(__doc__) 
    main(args)
