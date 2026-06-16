from dataclasses import dataclass, field
from typing import Annotated
import tyro
from ..utils.misc import get_device

@dataclass
class FlowMatchingTrainingConfig:
    """
    Config for training flow matching models.
    """
    seed: int = 0
    device: str = field(default_factory=get_device)

    cfm: str = 'cfm' # can be 'cfm', 'vp_cfm', 'ot_cfm', 'sb_cfm'
    
    # source and target distributions
    x0_dist: Annotated[str, tyro.conf.arg(aliases=("--source",))] = 'gaussian' # can be 'gaussian', '8gaussian', 'uniform', 'circle'
    x1_dist: Annotated[str, tyro.conf.arg(aliases=("--target",))] = 'gaussian' # can be 'gaussian', '8gaussian', 'uniform', 'circle'

    ### Below are fixed parameters by default
    batch_size: int = 1024
    lr: float = 1e-4
    epochs: Annotated[int, tyro.conf.arg(aliases=("--num-steps", "--num_steps"))] = 100000
    sigma: float = 0.0 # minimum variance in CFM
    width: int = 256 # width of the MLP

    overwrite: bool = False

    def __post_init__(self):
        if self.cfm == 'ot_cfm': # If batch size is too large, OT will be too expensive
            self.batch_size = min(self.batch_size, 256)

        slow_dist = ['moon', 's_curve', 'concentric_circle']
        if self.x0_dist in slow_dist or self.x1_dist in slow_dist: # sklearn-based datasets are slow to sample from
            self.batch_size = min(self.batch_size, 2048)

