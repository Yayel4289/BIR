import torch 
from torch import nn
import numpy as np 

from exps.points_classification.data.data_generation import generate_domains, get_train_val_loaders_from_source, get_train_val_loaders_from_source_target
from exps.points_classification.utils.point import Point
from src.metrics.losses import DANN_domain_loss, DANN_label_loss
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

    plots.plot_pts(points, show_target_label=True)
    plots.plot_decision_boundary(points, baseline)
    plots.show()

def run_dann(points: List[Point]) -> None:
    source_train_loader, source_val_loader, target_train_loader = get_train_val_loaders_from_source_target(points)

    dann = DANN(input_dim=2, output_dim=2)
    dann_trainer = DANN_Trainer(dann, source_train_loader, source_val_loader, target_train_loader, DANN_label_loss, DANN_domain_loss)
    dann_trainer.train_model()

    plots.plot_pts(points, show_target_label=True)
    plots.plot_decision_boundary(points, dann)
    plots.show()


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

points = generate_domains()
#run_baseline(points)
run_dann(points)
