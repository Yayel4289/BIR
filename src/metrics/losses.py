import torch

def PCIDA_domain_loss_function(domain_mean_var: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    domain_mean, domain_logvar = domain_mean_var[:, 0], domain_mean_var[:, 1]
    domain_logvar = torch.clamp(domain_logvar, min=1e-6)
    domain_loss = ((domain_mean - u)**2) / (2 * torch.exp(domain_logvar)) + 1/2 * domain_logvar
    return domain_loss.mean()

