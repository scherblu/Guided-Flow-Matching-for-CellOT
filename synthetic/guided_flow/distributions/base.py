import torch
from abc import ABC, abstractmethod



EPS = 1e-10


class BaseDistribution(ABC):
    """
    Base class for all distributions.
    """
    def __init__(self, energy_fn: str = None, energy_kwargs: dict = None):
        self.energy_fn_name = energy_fn or self.get_default_energy_fn_name()
        self.energy_kwargs = energy_kwargs or {}
        self._energy_fn_callable = None
        self.setup_energy_fn()

    def get_default_energy_fn_name(self) -> str:
        return None

    def set_energy(self, energy_fn: str, **kwargs):
        self.energy_fn_name = energy_fn
        self.energy_kwargs = kwargs
        self.setup_energy_fn()

    def setup_energy_fn(self):
        if self.energy_fn_name is None:
            self._energy_fn_callable = None
            return
        method = getattr(self, self.energy_fn_name, None)
        if method is not None and callable(method):
            self._energy_fn_callable = lambda x: method(x, **self.energy_kwargs)
        else:
            raise ValueError(f"Energy function '{self.energy_fn_name}' is not defined for {self.__class__.__name__}")

    @abstractmethod
    def sample(self, batch_size: int, device: str = None) -> torch.Tensor:
        """
        Returns: (B, 2)
        """
        pass

    @abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def get_J(self, x: torch.Tensor) -> torch.Tensor:
        if self._energy_fn_callable is None:
            raise NotImplementedError(
                f"No energy function configured for {self.__class__.__name__}. "
                f"Use set_energy() to set one."
            )
        return self._energy_fn_callable(x)

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_prob(x))


    def __str__(self):
        return self.__class__.__name__
    

def get_distribution(dist_name: str, energy_fn: str = None, **energy_kwargs) -> BaseDistribution:
    # import distributions here to avoid circular imports
    from guided_flow.distributions.uniform import UniformDistribution
    from guided_flow.distributions.laplace import LaplaceDistribution
    from guided_flow.distributions.gaussian import GaussianDistribution, EightGaussiansDistribution
    from guided_flow.distributions.circle import CircleDistribution, ConcentricCircleDistribution
    from guided_flow.distributions.moon import MoonDistribution
    from guided_flow.distributions.s_curve import SCurveDistribution
    from guided_flow.distributions.checkerboard import CheckerboardDistribution
    from guided_flow.distributions.spiral import SpiralDistribution

    if dist_name == 'gaussian':
        return GaussianDistribution(energy_fn=energy_fn, energy_kwargs=energy_kwargs)
    elif dist_name.startswith('gaussian_std'): # e.g. gaussian_std_0.1
        std = float(dist_name.split('_')[2])
        return GaussianDistribution(std=torch.ones(2) * std, energy_fn=energy_fn, energy_kwargs=energy_kwargs)
    elif dist_name == '8gaussian':
        return EightGaussiansDistribution(energy_fn=energy_fn, energy_kwargs=energy_kwargs)
    elif dist_name == 'uniform':
        return UniformDistribution(energy_fn=energy_fn, energy_kwargs=energy_kwargs)
    elif 'uniform_w' in dist_name:
        w = float(dist_name.split('_')[2])
        return UniformDistribution(low=-w, high=w, energy_fn=energy_fn, energy_kwargs=energy_kwargs)
    elif dist_name == 'laplace':
        return LaplaceDistribution(energy_fn=energy_fn, energy_kwargs=energy_kwargs)
    elif dist_name == 'circle':
        return CircleDistribution(energy_fn=energy_fn, energy_kwargs=energy_kwargs)
    elif dist_name == 'concentric_circle':
        return ConcentricCircleDistribution(energy_fn=energy_fn, energy_kwargs=energy_kwargs)
    elif dist_name == 'moon':
        return MoonDistribution(energy_fn=energy_fn, energy_kwargs=energy_kwargs)
    elif dist_name == 's_curve':
        return SCurveDistribution(energy_fn=energy_fn, energy_kwargs=energy_kwargs)
    elif dist_name == 'checkerboard':
        return CheckerboardDistribution(energy_fn=energy_fn, energy_kwargs=energy_kwargs)
    elif dist_name == 'spiral':
        return SpiralDistribution(energy_fn=energy_fn, energy_kwargs=energy_kwargs)
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")
    
