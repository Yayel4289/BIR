import pandas as pd

from exps.skinCancer.data_loader import get_X_y_tensors_from_df, get_train_val_test_loaders
from models.cida.cida import CIDA
import src.visualization.plots as plots
from src.training.trainers.cida_trainer import CIDA_Trainer
from src.training.lambda_config.progressive_lambda_config import ProgressiveLambdaConfig

source_train_loader, target_train_loader, source_val_loader, target_val_loader, source_test_loader, target_test_loader = get_train_val_test_loaders()

model = CIDA(input_dim=None, output_dim=7, convolutional=True)
trainer = CIDA_Trainer(
    model,
    source_train_loader, source_val_loader,
    target_train_loader, target_val_loader,
    lambda_config=ProgressiveLambdaConfig(4, 10)
)
trainer.train_model()

