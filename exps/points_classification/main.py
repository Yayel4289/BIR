"""
Usage:
    exps.points_classification.main -h
    exps.points_classification.main --plot-only
    exps.points_classification.main --model <model_name> [--hyperparam-selection]
    exps.points_classification.main [--comparison]

Options:
    -h                      Show this help
    --model <model_name>    Run <model_name> model. 
                            <model_name> can either be 'baseline' or 'dann'
    --plot-only             Show only points plot
    --hyperparam-selection  Perform an hyperparameter selection for gamma and max_lambda.
                            Only available for DANN
    --comparison            Compare models based on H-Divergence and F1 Score. Default behavior
                            (Hyperparameter selection for DANN)
"""

from docopt import docopt
from colorama import Fore, Style
import sys
import torch 
from torch import nn
import numpy as np 

from exps.points_classification.data.data_generation import generate_fused_domains, generate_intertwinning_moons, generate_split_domains, get_X_from_source_target, get_train_val_loaders_from_source, get_train_val_loaders_from_source_target
from exps.points_classification.utils.point import Point
from models.abstract_model import AbstractModel
from src.metrics.h_divergence import compute_empirical_h_divergence
from models.baseline.binary_classifier import BinaryClassifier
from models.dann.dann import DANN
import src.visualization.plots as plots
from src.training.trainers.baseline_trainer import BaselineTrainer
from src.training.trainers.dann_trainer import DANN_Trainer
from src.training.hyperparameter_selection import select_DANN_hyperparams

SEED = 42

def run_baseline(points: List[Point]) -> None:
    train_loader, val_loader = get_train_val_loaders_from_source(points)

    baseline = BinaryClassifier()
    baseline_trainer = BaselineTrainer(
        baseline, 
        train_loader, 
        val_loader, 
        nn.CrossEntropyLoss()
    )
    baseline_trainer.train_model()
    plot_model(points, baseline)

def run_dann(points: List[Point], hyperparam_selection: bool = False) -> None:
    source_train_loader, source_val_loader, target_train_loader, target_val_loader = get_train_val_loaders_from_source_target(points)

    kwargs = {}
    if hyperparam_selection:
        gamma, max_lambda = select_DANN_hyperparams(SEED, source_train_loader, source_val_loader, target_train_loader, target_val_loader)
        kwargs = {"gamma": gamma, "max_lambda": max_lambda}

    dann = DANN(input_dim=2, output_dim=2)
    dann_trainer = DANN_Trainer(
        dann, 
        source_train_loader, 
        source_val_loader, 
        target_train_loader, 
        target_val_loader,
        nn.CrossEntropyLoss(), 
        nn.BCEWithLogitsLoss(),
        **kwargs
    )
    dann_trainer.train_model()
    
    h_divergence = get_model_h_divergence(dann, points)
    print("H-Divergence:", h_divergence)
    plot_model(points, dann)

def make_table_comp(train_pts: List[Point]) -> None:
    pass


def get_model_h_divergence(model: AbstractModel, points: List[Point]) -> float:
    X_source_tensor, X_target_tensor = get_X_from_source_target(points)
    return compute_empirical_h_divergence(model, X_source_tensor, X_target_tensor)

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

def setup_seed() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def validate_model_params(formatted_model_name: str, hyperparam_selection: bool) -> bool: # True if model mode, else False
    if formatted_model_name is not None:
        if formatted_model_name not in ["baseline", "dann"]: 
            print_help_and_exit(Fore.RED + f"Wrong model name: '{formatted_model_name}'" + Style.RESET_ALL)
        elif formatted_model_name != "dann" and hyperparam_selection:
            print_help_and_exit(Fore.RED + f"Hyperparameter selection is only available for DANN" + Style.RESET_ALL)
    return True

def main() -> None:
    formatted_model_name = args["--model"].lower() if args["--model"] is not None else None
    hyperparam_selection = args["--hyperparam-selection"]
    model_mode = validate_model_params(formatted_model_name, hyperparam_selection)

    setup_seed()
    points = generate_intertwinning_moons()

    if args["--plot-only"]:
        plot_pts_only(points)
    elif model_mode:
        kwargs = {"hyperparam_selection": hyperparam_selection} if hyperparam_selection else {}
        run_func = globals().get("run_" + formatted_model_name)
        run_func(points, **kwargs)
    else:
        make_table_comp(points)

if __name__ == "__main__":
    args = docopt(__doc__)
    
    if args["-h"]: 
        print_help_and_exit()

    main()
