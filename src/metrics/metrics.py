from typing import Tuple
import torch
import numpy as np
from torchmetrics import F1Score
from exps.points_classification.data.data_generation import get_X_from_source_target, get_X_y_labels_from_source, get_X_y_labels_from_source_target, get_X_y_labels_from_target
from exps.points_classification.utils.point import Point
from models.abstract_model import AbstractModel
from models.dann.dann import DANN

# Eq. n°1 from DANN paper
def compute_empirical_h_divergence(model: DANN, X_source_tensor: torch.Tensor, X_target_tensor: torch.Tensor) -> float:
    n = X_source_tensor.shape[0]
    n_prime = X_target_tensor.shape[0]
    X = torch.cat([X_source_tensor, X_target_tensor], dim=0)
    
    model.eval()
    with torch.inference_mode():
        prediction_probs = model.predict_domain_prob(X)

    correct_source = (prediction_probs[:n]  < 0.5).long().sum() # Source (0)
    correct_target = (prediction_probs[n:] >= 0.5).long().sum() # Target (1)
    accuracy = (correct_source / n) + (correct_target / n_prime)
    h_divergence = 2 * (1 - accuracy)
    return h_divergence.item()

def compute_model_h_divergence(model: AbstractModel, points: List[Point]) -> float:
    X_source_tensor, X_target_tensor = get_X_from_source_target(points)
    return compute_empirical_h_divergence(model, X_source_tensor, X_target_tensor)

def compute_model_source_f1_score(model: AbstractModel, points: List[Point]) -> float:
    X_source_tensor, y_source_labels_tensor = get_X_y_labels_from_source(points)
    source_prediction_probs = model.predict_probs(X_source_tensor)
    source_predictions = (source_prediction_probs[:, 1] > source_prediction_probs[:, 0]).long().ravel()
    f1 = F1Score(task="binary")
    source_f1_score = f1(source_predictions, y_source_labels_tensor)
    return source_f1_score.item()

def compute_model_target_f1_score(model: AbstractModel, points: List[Point]) -> float:
    X_target_tensor, y_target_labels_tensor = get_X_y_labels_from_target(points)
    target_prediction_probs = model.predict_probs(X_target_tensor)
    target_predictions = (target_prediction_probs[:, 1] > target_prediction_probs[:, 0]).long().ravel()
    f1 = F1Score(task="binary")
    target_f1_score = f1(target_predictions, y_target_labels_tensor)
    return target_f1_score.item()

# Dict utils
def make_metrics_dict(h_divergence: bool=True):
    keys = ["source_f1_scores", "target_f1_scores"] + (["h_divergences"] if h_divergence else [])
    return {
        key: {
            "function": globals().get("compute_model_" + key[:-1]), # take away the 's',
            "list": [],
            "mean": 0,
            "std": 0
        } for key in keys
    }

def has_model_h_divergence(models_config, model_name):
    return models_config[model_name]["metrics"].get("h_divergences") is not None

def update_model_metrics_lists(model_metrics, model: AbstractModel, train_val_pts: List[Point]):
    for metric in model_metrics:
        computed_metric = model_metrics[metric]["function"](model, train_val_pts)
        model_metrics[metric]["list"].append(computed_metric)

def udpate_model_metrics_mean_std(model_metrics):
    for metric in model_metrics:
        model_metrics[metric]["mean"] = np.mean(model_metrics[metric]["list"])
        model_metrics[metric]["std"] = np.std(model_metrics[metric]["list"])

def get_table_string(models_config):
    name_space = len(max(models_config.keys(), key=len))
    rows = len(models_config)
    cols = 3
    cols_name_space = 15
    string =\
f"""
┌─{"─"*name_space}─┬─{"─"*cols_name_space}─┬─{"─"*cols_name_space}─┬─{"─"*cols_name_space}─┐
│ {"-":<{name_space}} │ Source F1 Score │ Target F1 Score │ H-Divergence    │
├─{"─"*name_space}─┼─{"─"*cols_name_space}─┼─{"─"*cols_name_space}─┼─{"─"*cols_name_space}─┤
"""
    for row_count, model_name in enumerate(models_config):
        string += f"│ {model_name:<{name_space}} │"
        for col_count, metric_details in enumerate(models_config[model_name]["metrics"].values()):
            mean, std = metric_details["mean"], metric_details["std"]
            string += f" {f"{mean:.3f} ± {std:.3f}":<{cols_name_space}} │"
        for _ in range(cols - col_count - 1):
            string += f" {"/":<{cols_name_space}} │"
        string += "\n"

        is_not_end = row_count < rows - 1
        if is_not_end:
            string += f"├─{"─"*name_space}─┼─{"─"*cols_name_space}─┼─{"─"*cols_name_space}─┼─{"─"*cols_name_space}─┤\n"
        else:
            string += f"└─{"─"*name_space}─┴─{"─"*cols_name_space}─┴─{"─"*cols_name_space}─┴─{"─"*cols_name_space}─┘"
    
    return string
            

