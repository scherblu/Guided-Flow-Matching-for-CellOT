import torch
import numpy as np
from sklearn.neighbors import KernelDensity as KDE

def compute_kl_divergence(target_dist, samples, scale=1.0):
    """
    Compute KL and reverse KL divergence between the target distribution 
    (P_target(x) \propto p(x) * e^(-scale * J(x))) and the generated samples.
    
    Args:
        target_dist: BaseDistribution object with log_prob and get_J methods.
        samples: torch.Tensor of shape (B, 2)
        scale: float, energy scale.
    
    Returns:
        kl: D_KL(P_samples || P_target)
        rev_kl: D_KL(P_target || P_samples)
    """
    device = samples.device
    samples_np = samples.detach().cpu().numpy()
    
    # Fit KDE to generated samples to get their log-density estimate
    # Bandwidth 0.1 is a reasonable heuristic for these 2D toy problems
    kde = KDE(bandwidth=0.1, kernel='gaussian')
    kde.fit(samples_np)
    
    # 1. Estimate log partition function Z for P_target
    # Z = \int p(x) e^(-scale * J(x)) dx \approx 1/N \sum e^(-scale * J(x_i)) where x_i ~ p(x)
    x1_samples = target_dist.sample(20000, device=device)
    energy = target_dist.get_J(x1_samples)
    log_z = torch.logsumexp(-scale * energy, dim=0) - torch.log(torch.tensor(20000.0, device=device))
    
    def get_log_p_target(x):
        return target_dist.log_prob(x) - scale * target_dist.get_J(x) - log_z

    # 2. Forward KL: D_KL(P_samples || P_target) = E_samples [ log P_samples - log P_target ]
    log_p_samples = torch.tensor(kde.score_samples(samples_np), device=device)
    log_p_target_at_samples = get_log_p_target(samples)
    kl = (log_p_samples - log_p_target_at_samples).mean().item()
    
    # 3. Reverse KL: D_KL(P_target || P_samples) = E_target [ log P_target - log P_samples ]
    # We use importance sampling from p(x) to estimate expectation under P_target
    # E_{P_target} [f(x)] = \int f(x) (p(x) e^-sJ / Z) dx = \int f(x) e^-sJ / Z p(x) dx \approx 1/N \sum f(x_i) e^-sJ_i / Z
    log_weights = -scale * energy - log_z
    weights = torch.exp(log_weights)
    weights = weights / weights.sum()
    
    log_p_samples_at_target = torch.tensor(kde.score_samples(x1_samples.detach().cpu().numpy()), device=device)
    log_p_target_at_target = get_log_p_target(x1_samples)
    
    rev_kl = (weights * (log_p_target_at_target - log_p_samples_at_target)).sum().item()
    
    return kl, rev_kl
