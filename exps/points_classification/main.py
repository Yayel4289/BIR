"""
Usage:
    exps.points_classification.main [--model <model_name>]
    exps.points_classification.main -h

Options:
    -h                      Show this help
    --model <model_name>    Run <model_name> model. 
                            <model_name> can either be 'baseline' (default) or 'dann'
"""

from docopt import docopt
from colorama import Fore, Style
import sys
import torch 
from torch import nn
import numpy as np 

from exps.points_classification.data.data_generation import generate_domains, get_train_val_loaders_from_source, get_train_val_loaders_from_source_target
from exps.points_classification.utils.point import Point
from models.baseline.binary_classifier import BinaryClassifier
from models.dann.dann import DANN
import src.visualization.plots as plots
from src.training.baseline_trainer import BaselineTrainer
from src.training.dann_trainer import DANN_Trainer

def run_baseline(points: List[Point]) -> None:
    train_loader, val_loader = get_train_val_loaders_from_source(points)

    baseline = BinaryClassifier()
    baseline_trainer = BaselineTrainer(baseline, train_loader, val_loader, nn.CrossEntropyLoss())
    baseline_trainer.train_model()
    plot_model(points, baseline)

def run_dann(points: List[Point]) -> None:
    source_train_loader, source_val_loader, target_train_loader = get_train_val_loaders_from_source_target(points)

    dann = DANN(input_dim=2, output_dim=2)
    dann_trainer = DANN_Trainer(dann, source_train_loader, source_val_loader, target_train_loader, nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss())
    dann_trainer.train_model()
    plot_model(points, dann)

def plot_model(points: List[Point], model: nn.Module):
    plots.plot_pts(points, show_target_label=True)
    plots.plot_decision_boundary(points, model)
    plots.show()

def print_help_and_exit(optional_msg=""):
    if optional_msg:
        print(optional_msg)
    print(__doc__)
    sys.exit(0)


if __name__ == "__main__":
    args = docopt(__doc__)
    
    if args["-h"]: print_help_and_exit()
    model_name = args["--model"]
    default_model = model_name is None
    run_func = None
    if not default_model:
        run_func = locals().get("run_" + model_name.lower())
        if run_func is None: 
            print_help_and_exit(Fore.RED + f"Wrong model name: '{model_name}'" + Style.RESET_ALL)

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    points = generate_domains()
    if model_name is None:
        run_baseline(points)
        sys.exit(0)

    run_func(points)
