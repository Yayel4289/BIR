import torch 
from torch import nn
import numpy as np 

from exps.points_classification.data.data_generation import generate_domains, get_train_test_loaders_from_domains
from models.base_model.binary_classifier import BinaryClassifier
import src.visualization.plots as plots
from src.training.base_trainer import BaseTrainer

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

points = generate_domains()
train_loader, test_loader = get_train_test_loaders_from_domains(points)

base_model = BinaryClassifier()
base_model_trainer = BaseTrainer(base_model, train_loader, test_loader, nn.BCELoss())

base_model_trainer.train_model()

plots.plot_pts(points, show_target_label=True)
plots.plot_decision_boundary(points, base_model)
plots.show()

