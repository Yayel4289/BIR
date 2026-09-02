from typing import Tuple
import torch
from torchmetrics import F1Score
from exps.points_classification.data.data_generation import get_X_from_source_target, get_X_y_labels_from_source, get_X_y_labels_from_source_target, get_X_y_labels_from_target
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

def get_model_h_divergence(model: AbstractModel, points: List[Point]) -> float:
    X_source_tensor, X_target_tensor = get_X_from_source_target(points)
    return compute_empirical_h_divergence(model, X_source_tensor, X_target_tensor)

def get_model_f1_score(model: AbstractModel, points: List[Point]) -> Tuple[float, float]:
    X_source_tensor, y_source_labels_tensor = get_X_y_labels_from_source(points)
    X_target_tensor, y_target_labels_tensor = get_X_y_labels_from_target(points)
    source_prediction_probs = model.predict_probs(X_source_tensor)
    target_prediction_probs = model.predict_probs(X_target_tensor)
    source_predictions = (source_prediction_probs[:, 1] > source_prediction_probs[:, 0]).long().ravel()
    target_predictions = (target_prediction_probs[:, 1] > target_prediction_probs[:, 0]).long().ravel()
    f1 = F1Score(task="binary")
    source_f1_score = f1(source_predictions, y_source_labels_tensor)
    target_f1_score = f1(target_predictions, y_target_labels_tensor)
    return source_f1_score.item(), target_f1_score.item()

