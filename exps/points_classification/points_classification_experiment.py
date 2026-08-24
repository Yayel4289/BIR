import torch 
from torch import nn
import numpy as np 

from exps.points_classification.data.data_generation import generate_domains, get_train_test_loaders_from_source
from models.baseline.binary_classifier import BinaryClassifier
import src.visualization.plots as plots
from src.training.baseline_trainer import BaselineTrainer

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

points = generate_domains()
train_loader, test_loader = get_train_test_loaders_from_source(points)

baseline = BinaryClassifier()
baseline_trainer = BaselineTrainer(baseline, train_loader, test_loader, nn.BCELoss())

baseline_trainer.train_model()

plots.plot_pts(points, show_target_label=True)
plots.plot_decision_boundary(points, baseline)
plots.show()

