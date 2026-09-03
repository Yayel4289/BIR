"""
Usage:
    exps.points_classification.main [-h]
    exps.points_classification.main --plot-only
    exps.points_classification.main --model <model_name> [--hyperparam-selection]
    exps.points_classification.main --comparison

Options:
    -h                      Show this help. Default behavior is no parameter is passed
    --model <model_name>    Run <model_name> model. 
                            <model_name> can either be 'baseline' or 'dann'
    --plot-only             Show only points plot
    --hyperparam-selection  Perform an hyperparameter selection for gamma and max_lambda.
                            Only available for DANN
    --comparison            Compare models based on H-Divergence and F1 Score.
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
from src.metrics.metrics import compute_model_h_divergence, compute_model_source_f1_score, compute_model_target_f1_score, get_table_string, has_model_h_divergence, make_metrics_dict, udpate_model_metrics_mean_std, update_model_metrics_lists
from models.baseline.binary_classifier import BinaryClassifier
from models.dann.dann import DANN
import src.visualization.plots as plots
from src.training.trainers.baseline_trainer import BaselineTrainer
from src.training.trainers.dann_trainer import DANN_Trainer
from src.training.hyperparameter_selection import select_DANN_hyperparams

DEFAULT_SEED = 42
COMPARISON_SEEDS = [DEFAULT_SEED, 89, 0]

def run_baseline(points: List[Point], plot: bool = True, verbose: bool = True) -> BinaryClassifier:
    train_loader, val_loader = get_train_val_loaders_from_source(points)

    baseline = BinaryClassifier()
    baseline_trainer = BaselineTrainer(
        baseline, 
        train_loader, 
        val_loader, 
        nn.CrossEntropyLoss(),
        verbose=verbose
    )
    baseline_trainer.train_model()
    if verbose:
        source_f1_score, target_f1_score = compute_model_source_f1_score(baseline, points), compute_model_target_f1_score(baseline, points)
        print("Source F1 Score:", source_f1_score)
        print("Target F1 Score:", target_f1_score)
    if plot:
        plot_model(points, baseline)
    return baseline

def run_dann(points: List[Point], hyperparam_selection: bool = False, plot: bool = True, verbose: bool = True) -> DANN:
    source_train_loader, source_val_loader, target_train_loader, target_val_loader = get_train_val_loaders_from_source_target(points)

    kwargs = {}
    if hyperparam_selection:
        gamma, max_lambda = select_DANN_hyperparams(
            DEFAULT_SEED, 
            source_train_loader, 
            source_val_loader, 
            target_train_loader, 
            target_val_loader, 
            verbose=verbose
        )
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
        verbose=verbose,
        **kwargs
    )
    dann_trainer.train_model()
    
    if verbose:
        source_f1_score, target_f1_score = compute_model_source_f1_score(dann, points), compute_model_target_f1_score(dann, points)
        h_divergence = compute_model_h_divergence(dann, points)
        print("Source F1 Score:", source_f1_score)
        print("Target F1 Score:", target_f1_score)
        print("H-Divergence:", h_divergence)
    if plot:
        plot_model(points, dann)
    return dann

def make_table_comparison() -> None:
    models_config =  {
        "Baseline": {
            "run_func": run_baseline,
            "metrics": make_metrics_dict(h_divergence=False),
            "kwargs": {
                "verbose": False,
                "plot": False
            }
        },
        "DANN": {
            "run_func": run_dann,
            "metrics": make_metrics_dict(),
            "kwargs": {
                "hyperparam_selection": True,
                "verbose": False,
                "plot": False
            }
        }
    }
    for i, seed in enumerate(COMPARISON_SEEDS):
        setup_seed(seed)
        train_val_pts = generate_intertwinning_moons()
        test_pts = generate_intertwinning_moons()
        for model_name in models_config:
            run_func = models_config[model_name]["run_func"]
            kwargs = models_config[model_name]["kwargs"]
            model = run_func(train_val_pts, **kwargs)

            model_metrics = models_config[model_name]["metrics"]
            update_model_metrics_lists(model_metrics, model, train_val_pts)
            if i == len(COMPARISON_SEEDS) - 1: # last pass
                udpate_model_metrics_mean_std(model_metrics)

    print(get_table_string(models_config))

 
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

def validate_model_params(formatted_model_name: str, hyperparam_selection: bool) -> bool: # True if model mode, else False
    if formatted_model_name is None:
        return False
    if formatted_model_name not in ["baseline", "dann"]: 
        print_help_and_exit(Fore.RED + f"Wrong model name: '{formatted_model_name}'" + Style.RESET_ALL)
    elif formatted_model_name != "dann" and hyperparam_selection:
        print_help_and_exit(Fore.RED + f"Hyperparameter selection is only available for DANN" + Style.RESET_ALL)
    return True


def main() -> None:
    if args["-h"]: 
        print_help_and_exit()
    if args["--comparison"]:
        make_table_comparison()
        sys.exit(0)

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
        print_help_and_exit()

if __name__ == "__main__":
    args = docopt(__doc__) 
    main()
