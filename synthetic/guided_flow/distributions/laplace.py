import math
import torch
from .base import BaseDistribution
from ..utils.misc import get_device

class LaplaceDistribution(BaseDistribution):
    def __init__(self, mu: torch.Tensor = torch.zeros(2), b: torch.Tensor = torch.eye(2), energy_fn: str = None, energy_kwargs: dict = None):
        super().__init__(energy_fn=energy_fn, energy_kwargs=energy_kwargs)
        self.mu = mu
        self.b = b

    def get_default_energy_fn_name(self) -> str:
        return 'target_point_quadratic_distance'

    def target_point_quadratic_distance(self, x: torch.Tensor, target: list = [1.0, 1.0]) -> torch.Tensor:
        target_tensor = torch.tensor(target, dtype=x.dtype, device=x.device)
        return torch.sum((x - target_tensor) ** 2, dim=-1)

    def sample(self, batch_size: int, device: str = None) -> torch.Tensor:
        """
        Sample from a 2D multivariate Laplace distribution using PyTorch.

        Args:
            mean (torch.Tensor): Mean vector of shape (B, 2).
            covariance (torch.Tensor): Covariance matrix of shape (2, 2).
            num_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Samples of shape (num_samples, 2).
        """
        if device is None:
            device = get_device()

        # Ensure mean and covariance have the correct shape
        mean = self.mu.view(2).to(device=device)
        covariance = self.b.to(device=device)

        # Perform Cholesky decomposition for covariance (on CPU as fallback for MPS)
        L = torch.linalg.cholesky(covariance.cpu()).to(device)

        # Sample from univariate Laplace distribution
        u = torch.rand(batch_size, 2, device=device) - 0.5
        # Clamp u to avoid log of zero or negative
        u = torch.clamp(u, min=-0.4999, max=0.4999)
        univariate_samples = -torch.sign(u) * torch.log(1 - 2 * torch.abs(u))

        # Transform samples to match the multivariate distribution
        multivariate_samples = mean + univariate_samples @ L.T

        return multivariate_samples
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x (torch.Tensor): Samples of shape (B, 2).
        """
        # Ensure shapes
        mean = self.mu.view(2).to(device=x.device)
        covariance = self.b.to(device=x.device)
        x = x.view(-1, 2).to(device=x.device)
        
        # Compute covariance inverse and determinant (on CPU as fallback for MPS)
        cov_inv = torch.linalg.inv(covariance.cpu()).to(device=x.device)
        cov_det = torch.det(covariance.cpu()).to(device=x.device)
        
        # Compute Mahalanobis distance
        diff = x - mean
        mahalanobis = torch.sqrt(torch.sum(diff @ cov_inv * diff, dim=1))
        
        # Compute log probability
        log_normalizer = -0.5 * torch.log(cov_det) - torch.log(torch.tensor(2.0, device=x.device))
        log_prob = log_normalizer - mahalanobis
        
        return log_prob
