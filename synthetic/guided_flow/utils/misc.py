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


def print_model_summary(model, example_inputs=None):
    """Prints a beautiful summary card of a PyTorch model, including layers, shapes, parameters, and size."""
    import torch
    
    summary_data = []
    hooks = []
    
    def register_hook(module):
        def hook(module, input, output):
            # Resolve input shape
            if isinstance(input, tuple):
                in_shape = [list(i.shape) if hasattr(i, 'shape') else str(type(i)) for i in input]
                if len(in_shape) == 1:
                    in_shape = in_shape[0]
            else:
                in_shape = list(input.shape) if hasattr(input, 'shape') else str(type(input))
            
            # Resolve output shape
            if hasattr(output, 'shape'):
                out_shape = list(output.shape)
            elif isinstance(output, tuple):
                out_shape = [list(o.shape) if hasattr(o, 'shape') else str(type(o)) for o in output]
            else:
                out_shape = str(type(output))
                
            num_params = sum(p.numel() for p in module.parameters(recurse=False))
            trainable = any(p.requires_grad for p in module.parameters(recurse=False))
            
            summary_data.append({
                'name': getattr(module, '_summary_name', module.__class__.__name__),
                'type': module.__class__.__name__,
                'input_shape': in_shape,
                'output_shape': out_shape,
                'params': num_params,
                'trainable': trainable
            })
            
        # Register hook on leaf modules, Sequentials, and the root model
        if len(list(module.children())) == 0 or isinstance(module, torch.nn.Sequential) or module is model:
            hooks.append(module.register_forward_hook(hook))
            
    # Annotate modules with names
    for name, module in model.named_modules():
        module._summary_name = name if name else 'model'
        
    model.apply(register_hook)
    
    # Perform forward pass if example inputs are provided
    if example_inputs is not None:
        is_training = model.training
        model.eval()
        with torch.no_grad():
            if isinstance(example_inputs, tuple):
                model(*example_inputs)
            elif isinstance(example_inputs, dict):
                model(**example_inputs)
            else:
                model(example_inputs)
        model.train(is_training)
                
    # Remove hooks
    for h in hooks:
        h.remove()
        
    # Clean up names
    for _, module in model.named_modules():
        if hasattr(module, '_summary_name'):
            del module._summary_name
            
    # Print the model summary card
    print('=' * 105)
    print(f'Model Summary: {model.__class__.__name__}')
    print('=' * 105)
    print(f'   {"Layer (type)":<40} | {"Input Shape":<22} | {"Output Shape":<22} | {"Param #":<12}')
    print('-' * 105)
    for layer in summary_data:
        if layer['name'] == 'model' and len(summary_data) > 1:
            continue
        name_str = f"   {layer['name']} ({layer['type']})"
        in_shape_str = str(layer['input_shape'])
        out_shape_str = str(layer['output_shape'])
        param_str = f"{layer['params']:,}" if layer['params'] > 0 else '--'
        print(f'{name_str:<43} | {in_shape_str:<22} | {out_shape_str:<22} | {param_str:<12}')
        
    print('-' * 105)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    
    print(f'   Total Params:         {total_params:,}')
    print(f'   Trainable Params:     {trainable_params:,}')
    print(f'   Non-Trainable Params: {non_trainable_params:,}')
    print(f'   Model Size:           {model_size_mb:.3f} MB')
    print('=' * 105)

