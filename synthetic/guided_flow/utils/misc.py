import os

import yaml


def get_device():
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        # Set fallback for missing operators on MPS (like Cholesky)
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        return 'mps'
    else:
        return 'cpu'

def deterministic(seed):
    import torch
    import numpy
    import random
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        # MPS doesn't have a separate manual_seed_all, torch.manual_seed is sufficient
        pass
    numpy.random.seed(seed)
    random.seed(seed)
    # torch deterministic algorithm
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_device(cfg, outmost=True, debug=False):
    """Detect device if 'auto', set environment variable CUDA_VISIBLE_DEVICES if applicable, 
    then set cfg.device and cfg.*.device recursively"""
    import torch
    
    # Resolve 'auto' device
    if hasattr(cfg, 'device') and cfg.device == 'auto':
        cfg.device = get_device()

    # cuda visible device
    if outmost and torch.cuda.is_available() and 'cuda' in cfg.device:
        import os
        if ':' in cfg.device:
            cuda_num = cfg.device.split(':')[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_num
            cfg.device = 'cuda:0' # After setting VISIBLE_DEVICES, we usually use cuda:0
    
    # recursively set device
    for key, value in cfg.__dict__.items():
        if key == 'device' and value == 'auto':
             setattr(cfg, key, get_device())

        if hasattr(value, '__dict__'):
            setup_device(value, outmost=False)

def save_config(cfg, exp_dir, config_name='config.yaml'):
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
    else:
        print(f"[ Initialize Logger ] Will save model to {exp_dir}, old cps will be overwritten!")
    with open(os.path.join(exp_dir, config_name), 'w') as f:
        yaml.dump(vars(cfg), f)
