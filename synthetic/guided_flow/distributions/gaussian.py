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

    def get_J_(self, x1: torch.Tensor) -> torch.Tensor:
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
    
    # get_J_continuous_single_blob
    def get_J(self, x1: torch.Tensor, target_blob_idx: int = 0, std_target:float =0.1) -> torch.Tensor:
        """
        Continuous energy function for a single Gaussian blob. 
        best for gradient-based guidance methods

        Args: 
            x1 (B, 2): Generated samples
            target_blob_idx: An integer from 0 to 7 representing the blob to keep
        Returns: 
            J (B,): Smooth continuous energy values
        """
        # Get the coordinate of the target center, e.g., self.centers[0]
        target_center = self.centers[target_blob_idx].to(x1.device) # Shape: (2,)
        
        # Compute squared Euclidean distance to that specific center
        sq_dist = torch.sum((x1 - target_center) ** 2, dim=-1)      # Shape: (B,)
        
        # Scale by the cluster variance (the std in your code is 0.1, so std^2 = 0.01)
        return sq_dist / (2 * (std_target ** 2))
    
    def get_J_three_blobs(self, x1: torch.Tensor) -> torch.Tensor:
        """
        Args: 
            x1 (B, 2): Generated samples
        Returns: 
            J (B,): Energy values (0 for blobs 0, 2, 5; 20 for others)
        """
        # 1. Compute distances to all 8 centers
        dists = torch.cdist(x1, self.centers.to(x1.device)) # Shape: (B, 8)
        closest_idx = torch.argmin(dists, dim=1)            # Shape: (B,)
        
        # 2. Check if the closest index is 0, 2, or 5
        mask = (closest_idx == 0) | (closest_idx == 2) | (closest_idx == 5)
        
        # 3. Assign 0 energy to targets, 20 to everything else
        J = torch.where(mask, 0.0, 20.0)
        return J

    def get_J_continuous_three_blobs(self, x1: torch.Tensor, target_indices: list = [0, 2, 5], std_target:float =0.1) -> torch.Tensor:
        """
        Args: 
            x1 (B, 2): Generated samples
        Returns: 
            J (B,): Smooth multi-well energy values
        """
        # 1. Extract the coordinates for centers 0, 2, and 5
        target_centers = self.centers[target_indices].to(x1.device) # Shape: (3, 2)
        
        # 2. Compute squared distances to these 3 centers for all batch items
        # x1 unsqueezed to (B, 1, 2), target_centers unsqueezed to (1, 3, 2)
        sq_dists = torch.sum((x1.unsqueeze(1) - target_centers.unsqueeze(0)) ** 2, dim=-1) # Shape: (B, 3)
        
        # 3. Compute the individual negative exponents (-dist^2 / 2*sigma^2)
        # Given std = 0.1, 2 * variance = 2 * (0.1^2) = 0.02
        exponents = -sq_dists / (2 * (std_target ** 2)) # Shape: (B, 3)
        
        # 4. Use LogSumExp for numerical stability to get the final smooth negative log landscape
        J = -torch.logsumexp(exponents, dim=-1) # Shape: (B,)
        return J

    def __name__(self):
        return f'8 Gaussians'
