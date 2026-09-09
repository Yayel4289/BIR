import pandas as pd
from torch import nn

from exps.skinCancer.data_loader import get_X_y_tensors_from_df, get_train_val_test_loaders
from models.baseline.classifier import Classifier
from models.dann.dann import DANN
from src.metrics.metrics import compute_model_f1_score_from_loader, compute_model_h_divergence
from models.cida.cida import CIDA
import src.visualization.plots as plots
from src.training.trainers.cida_trainer import CIDA_Trainer
from src.training.lambda_config.progressive_lambda_config import ProgressiveLambdaConfig
from src.training.trainers.dann_trainer import DANN_Trainer
from src.training.trainers.baseline_trainer import BaselineTrainer

source_train_loader, target_train_loader, source_val_loader, target_val_loader, source_test_loader, target_test_loader = get_train_val_test_loaders()

model = DANN(input_dim=None, output_dim=7, convolutional=True)
trainer = DANN_Trainer(
    model,
    source_train_loader, source_val_loader,
    target_train_loader, target_val_loader,
    lambda_config=ProgressiveLambdaConfig(3, 10)
)
trainer.train_model()

source_f1_score = compute_model_f1_score_from_loader(model, source_test_loader, task="multiclass", num_classes=7)
target_f1_score = compute_model_f1_score_from_loader(model, target_test_loader, task="multiclass", num_classes=7)
h_divergence = compute_model_h_divergence(model, source_test_loader, target_test_loader)
print("Source F1 Score:", source_f1_score)
print("Target F1 Score:", target_f1_score)
print("H-Divergence:", h_divergence)

