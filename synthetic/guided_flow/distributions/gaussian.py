from .base import BaseDistribution
from ..utils.misc import get_device
import torch
import math

class GaussianDistribution(BaseDistribution):
    def __init__(
        self, 
        mean: torch.Tensor = torch.zeros(2), 
        std: torch.Tensor = torch.ones(2)
    ):
        super().__init__()
        self.mean = mean
        self.std = std

    def sample(self, batch_size: int, device: str = None) -> torch.Tensor:
        if device is None:
            device = get_device()
        eps = torch.randn(batch_size, 2, device=device)
        return eps * self.std.to(device) + self.mean.to(device)
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * ((x - self.mean.to(x.device)) / self.std.to(x.device)).pow(2).sum(-1) - torch.log(self.std.to(x.device)).sum(-1) - 0.5 * math.log(2 * torch.pi)

    def get_J(self) -> torch.Tensor:
        raise NotImplementedError("J for gaussian is not implemented yet")

    def __name__(self):
        return f'Gaussian'

class EightGaussiansDistribution(BaseDistribution):
    def __init__(
        self, 
        centers: torch.Tensor = torch.tensor(
            [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1.0 / math.sqrt(2), 1.0 / math.sqrt(2)),
                (1.0 / math.sqrt(2), -1.0 / math.sqrt(2)),
                (-1.0 / math.sqrt(2), 1.0 / math.sqrt(2)),
                (-1.0 / math.sqrt(2), -1.0 / math.sqrt(2)),
            ], dtype=torch.float32
        ), 
        std: torch.Tensor = torch.ones(8, 2) * 0.1
    ):
        super().__init__()
        self.centers = centers
        self.std = std

    def sample(self, batch_size: int, device: str = None) -> torch.Tensor:
        if device is None:
            device = get_device()
        multi = torch.multinomial(torch.ones(8, device=device), batch_size, replacement=True) # size (batch_size,)
        eps = torch.randn(batch_size, 2, device=device)
        centers = self.centers.to(device).repeat(batch_size, 1, 1)[torch.arange(batch_size, device=device), multi]
        stds = self.std.to(device).repeat(batch_size, 1, 1)[torch.arange(batch_size, device=device), multi]
        return centers + eps * stds
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.prob(x).clamp(1e-10).log()

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Approximate with the PDF of 8 non-overlapping gaussians.
        """
        one_log_prob = -0.5 * ((x.unsqueeze(1) - self.centers.to(x.device).unsqueeze(0)) / self.std.to(x.device).unsqueeze(0)).pow(2).sum(-1) # (B, 8, 2) -> (B, 8)
        one_log_prob_z = torch.log(self.std.to(x.device)).sum(-1) - 0.5 * math.log(2 * torch.pi) # (B, 8)
        prob = (torch.exp(one_log_prob) / torch.exp(one_log_prob_z)).mean(-1)
        return prob

    def get_J(self, x1: torch.Tensor) -> torch.Tensor:
        """
        discrete masking for energy function J of 8gaussian
        exponentially weight the index 0 to 7
        best for Exact Monte Carlo Guidance

        Args: x1 (B, 2)
        Returns: J (B,)
        """
        # compute distances between each sample and the 8 centers
        dists = torch.cdist(x1, self.centers.to(x1.device)) # (B, 8)
        # return the index of the closest center for each sample (B,)
        return torch.argmin(dists, dim=1)
    
    def get_J_continuous_single_blob(self, x1: torch.Tensor, target_blob_idx: int, std_target:float =0.1) -> torch.Tensor:
        """
        Continuous energy function for a single Gaussian blob. 
        best for gradient-based guidance methods

        Args: 
            x1 (B, 2): Generated samples
            target_blob_idx: An integer from 0 to 7 representing the blob to keep
        Returns: 
            J (B,): Smooth continuous energy values
        """
        # Get the coordinate of the target center, e.g., self.centers[3]
        target_center = self.centers[target_blob_idx].to(x1.device) # Shape: (2,)
        
        # Compute squared Euclidean distance to that specific center
        sq_dist = torch.sum((x1 - target_center) ** 2, dim=-1)      # Shape: (B,)
        
        # Scale by the cluster variance (the std in your code is 0.1, so std^2 = 0.01)
        return sq_dist / (2 * (std_target ** 2))

    def __name__(self):
        return f'8 Gaussians'
