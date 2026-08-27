import torch
from models.dann.dann import DANN

# Eq. n°1 from DANN paper
def compute_empirical_h_divergence(model: DANN, X_source_tensor: torch.Tensor, X_target_tensor: torch.Tensor) -> float:
    n = X_source_tensor.shape[0]
    n_prime = X_target_tensor.shape[0]
    X = torch.cat([X_source_tensor, X_target_tensor], dim=0)
    y = torch.cat([
        torch.zeros(n, dtype=torch.long),
        torch.ones(n_prime, dtype=torch.long)
    ])
    
    model.eval()
    with torch.inference_mode():
        prediction_probs = model.predict_domain_prob(X)
        print(f"Min prediction: {prediction_probs.min().item()}")
        print(f"Max prediction: {prediction_probs.max().item()}")
        print(f"Mean prediction: {prediction_probs.mean().item()}")

    correct_source = (prediction_probs[:n]  < 0.5).float().sum() # Source (0)
    correct_target = (prediction_probs[n:] >= 0.5).float().sum() # Target (1)
    accuracy = (correct_source / n) + (correct_target / n_prime)
    h_divergence = 2 * (1 - accuracy)
    return h_divergence.item()

