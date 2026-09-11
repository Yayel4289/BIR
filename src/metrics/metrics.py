from typing import Dict, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics import F1Score
from exps.points_classification.data.data_generation import get_X_from_source_target, get_X_y_labels_from_source, get_X_y_labels_from_source_target, get_X_y_labels_from_target
from exps.points_classification.utils.point import Point
from models.abstract_model import AbstractModel
from models.dann.dann import DANN
from src.utils.utils import to_device

# Eq. n°1 from DANN paper
def compute_model_h_divergence(model: DANN, source_test_loader: DataLoader, target_test_loader: DataLoader, device: torch.device = None) -> float:
    n = len(source_test_loader.dataset)
    n_prime = len(target_test_loader.dataset)
    correct_source = 0
    correct_target = 0
    model.eval()
    with torch.inference_mode():
        for (X_source_test_batch, _) in source_test_loader:
            X_source_test_batch = to_device(device, X_source_test_batch)
            prediction_probs = model.predict_domain_prob(X_source_test_batch)
            correct_source += (prediction_probs  < 0.5).long().sum().item()
        for (X_target_test_batch, _) in target_test_loader:
            X_target_test_batch = to_device(device, X_target_test_batch)
            prediction_probs = model.predict_domain_prob(X_target_test_batch)
            correct_target += (prediction_probs >= 0.5).long().sum().item()
    accuracy = (correct_source / n) + (correct_target / n_prime)
    h_divergence = 2 * (1 - accuracy)
    return h_divergence

def compute_model_source_f1_score(model: AbstractModel, source_test_loader: DataLoader, target_test_loader: DataLoader, device: torch.device = None) -> float:
    return compute_model_f1_score_from_loader(model, source_test_loader, device=device)

def compute_model_target_f1_score(model: AbstractModel, source_test_loader: DataLoader, target_test_loader: DataLoader, device: torch.device = None) -> float:
    return compute_model_f1_score_from_loader(model, target_test_loader, device=device)

def compute_model_f1_score_from_loader(model: AbstractModel, test_loader: DataLoader, device: torch.device = None) -> float:
    model.eval()
    f1 = F1Score(task="binary", average="macro")
    with torch.inference_mode():
        for (X_test_batch, y_test_batch) in test_loader:
            X_test_batch, y_test_batch = to_device(device, X_test_batch, y_test_batch)
            y_test_batch_labels = y_test_batch[:, 0]
            prediction_probs = model.predict_probs(X_test_batch)

            predictions = torch.argmax(prediction_probs, dim=1)
            f1.update(predictions, y_test_batch_labels)
    f1.compute()
    return f1._computed.item()

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

def update_model_metrics_lists(
        model_metrics, 
        model: AbstractModel, 
        source_test_loader: DataLoader, 
        target_test_loader: DataLoader, 
        device: torch.device
    ) -> None:
    for metric in model_metrics:
        computed_metric = model_metrics[metric]["function"](model, source_test_loader, target_test_loader, device=device)
        model_metrics[metric]["list"].append(computed_metric)

def udpate_model_metrics_mean_std(model_metrics):
    for metric in model_metrics:
        model_metrics[metric]["mean"] = np.mean(model_metrics[metric]["list"])
        model_metrics[metric]["std"] = np.std(model_metrics[metric]["list"])

def get_model_metrics(models_config) -> str:
    res = ""
    for model_name in models_config: # only one
        res += model_name + ":\n"
        for metric_name, metric_details in models_config[model_name]["metrics"].items():
            mean, std = metric_details["mean"], metric_details["std"]
            res += f"{metric_name}: {mean:.3f} ± {std:.3f}\n"
    return res

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
            string += f" {mean:.3f} ± {std:.3f}".ljust(cols_name_space + 1) + "│"
        for _ in range(cols - col_count - 1):
            string += f" {'/':<{cols_name_space}} │"
        string += "\n"

        is_not_end = row_count < rows - 1
        if is_not_end:
            string += f"├─{'─' * name_space}─┼─{'─' * cols_name_space}─┼─{'─' * cols_name_space}─┼─{'─' * cols_name_space}─┤\n"
        else:
            string += f"└─{'─' * name_space}─┴─{'─' * cols_name_space}─┴─{'─' * cols_name_space}─┴─{'─' * cols_name_space}─┘"
    
    return string
            

