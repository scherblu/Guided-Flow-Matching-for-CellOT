import math
from .base import BaseDistribution, EPS
from ..utils.misc import get_device
import torch
import sklearn

class CircleDistribution(BaseDistribution):
    def __init__(self, r: float = 1, sigma: float = 0.05, energy_fn: str = None, energy_kwargs: dict = None):
        super().__init__(energy_fn=energy_fn, energy_kwargs=energy_kwargs)
        self.r = r
        self.sigma = sigma

    def get_default_energy_fn_name(self) -> str:
        return 'shifted_circle_boundary_distance'

    def shifted_circle_boundary_distance(self, x: torch.Tensor, center: list = [1.0, 1.0], radius: float = 1.0, scale: float = 5.0, clamp_max: float = 10.0) -> torch.Tensor:
        center_tensor = torch.tensor(center, dtype=x.dtype, device=x.device)
        dist_sq_minus_r_sq = (x - center_tensor).pow(2).sum(-1) - radius**2
        return (dist_sq_minus_r_sq.abs() * scale).clamp(0, clamp_max)

    def sample(self, batch_size: int, device: str = None) -> torch.Tensor:
        if device is None:
            device = get_device()
        theta = torch.rand(batch_size, device=device) * 2 * torch.pi
        r = self.r + torch.randn(batch_size, device=device) * self.sigma
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.stack([x, y], dim=-1)
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        print('here')
        l = 2 * torch.pi * self.r
        r_x = (x[:, 0] ** 2 + x[:, 1] ** 2).sqrt() # (B,)
        log_p_gaussian_along_r = -0.5 * torch.log(2 * torch.pi * self.sigma ** 2) - (r_x - self.r) ** 2 / 2 / self.sigma ** 2 # (B,)
        return log_p_gaussian_along_r - math.log(l)
    
    def prob(self, x: torch.Tensor) -> torch.Tensor:
        l = 2 * torch.pi * self.r
        r_x = (x[:, 0] ** 2 + x[:, 1] ** 2).sqrt() # (B,)
        p_gaussian_along_r = 1 / math.sqrt(2 * torch.pi * self.sigma ** 2) * torch.exp(-(r_x - self.r) ** 2 / 2 / self.sigma ** 2) # (B,)
        return p_gaussian_along_r / l

    def __name__(self):
        return f'Circle'
        

class ConcentricCircleDistribution(BaseDistribution):
    def __init__(self, r: float = 0.5, sigma: float = 0.05, energy_fn: str = None, energy_kwargs: dict = None):
        super().__init__(energy_fn=energy_fn, energy_kwargs=energy_kwargs)
        self.r = r
        self.sigma = sigma

    def get_default_energy_fn_name(self) -> str:
        return 'unit_circle_boundary_distance'

    def unit_circle_boundary_distance(self, x: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
        dist_sq = torch.sum(x ** 2, dim=-1)
        return (dist_sq - radius**2).abs().clamp(0)

    def sample(self, batch_size: int, device: str = None) -> torch.Tensor:
        if device is None:
            device = get_device()
        samples, _ = sklearn.datasets.make_circles(n_samples=batch_size, noise=self.sigma, factor=self.r) # (B, 2)
        return torch.tensor(samples, device=device, dtype=torch.float32)
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Probability for concentric circles is complicated, not implemented yet")
    
    def prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Probability for concentric circles is complicated, not implemented yet")
