from .base import BaseDistribution, EPS
from ..utils.misc import get_device
import torch

class CheckerboardDistribution(BaseDistribution):
    """
    The distribution is a checkerboard with n x m cells, but only about n * m / 2 cells 
    have nonzero probability. Inside each cell, the probability is uniform.
    Cells that are adjacent to each other are not simultaneously non-zero.
    """
    def __init__(self, n: int = 4, m: int = 4, size: float = 1, device: str = None, energy_fn: str = None, energy_kwargs: dict = None):
        super().__init__(energy_fn=energy_fn, energy_kwargs=energy_kwargs)
        self.n = n
        self.m = m
        self.size = size
        self.device = device if device is not None else get_device()

    def get_default_energy_fn_name(self) -> str:
        return 'alternating_stripe_indicator'

    def alternating_stripe_indicator(self, x: torch.Tensor, scale: float = 10.0) -> torch.Tensor:
        # Determine which cell the point is in
        n = (x[:, 1] * 2 / (self.size)).floor()
        
        # Determine which cell the point is in
        # so white or black
        return (n % 2 == 0).float() * scale

    def sample(self, batch_size: int, device: str = None) -> torch.Tensor:
        if device is None:
            device = self.device
        # choose which cells are non-zero
        n = torch.randint(0, self.n, (batch_size, 1), device=device)
        m = torch.randint(0, (self.m + 1) // 2, (batch_size, 1), device=device) * 2 + n % 2
        # sample from the chosen cells
        x = torch.rand(batch_size, 2, device=device) * self.size
        x = x + torch.cat([n, m], dim=-1) * self.size - torch.tensor([self.n / 2, self.m / 2], device=device, dtype=torch.float32) * self.size
        return x / torch.tensor([self.n, self.m], device=device, dtype=torch.float32) / self.size * 2

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(self.prob(x))
    
    def prob(self, x: torch.Tensor) -> torch.Tensor:
        is_in_cell = self.get_J(x) / 10
        return is_in_cell / (self.n * self.m / 2) / self.size / self.size
