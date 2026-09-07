import torch

def PCIDA_domain_loss_function(domain_mean_var: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    domain_mean, domain_var = domain_mean_var[:, 0], domain_mean_var[:, 1]
    domain_var = torch.clamp(domain_var, min=1e-6)
    domain_loss = ((domain_mean - u)**2) / (2 * domain_var) + 1/2 * torch.log(domain_var)
    return domain_loss.mean()

