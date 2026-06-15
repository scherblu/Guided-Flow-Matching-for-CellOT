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


def check_energy_compatibility(model_path, distribution, current_scale=None):
    """
    Verifies that the energy function configured on the distribution matches
    what the model was trained on, by looking up the local YAML config or checking WandB logs.
    Throws ValueError if there is a mismatch.
    """
    import os
    import yaml
    import json
    import re

    model_dir = os.path.dirname(model_path)
    model_filename = os.path.basename(model_path)
    train_cfg = None
    is_fallback_cfg = False

    # 1. First, check for a 1-to-1 matching config file (same timestamp/name)
    matching_config_path = model_path.replace('.pth', '.yaml')
    if not os.path.exists(matching_config_path):
        matching_config_path = model_path.replace('.pth', '.yml')

    if os.path.exists(matching_config_path):
        try:
            with open(matching_config_path, 'r') as f:
                train_cfg = yaml.safe_load(f)
            print(f"[ Verification ] Loaded training config from {matching_config_path}")
        except Exception as e:
            print(f"Warning: Failed to load training config {matching_config_path}: {e}")

    # Fallback to the old general scale-based config if matching config wasn't found
    if not train_cfg:
        scale_match = re.search(r'scale_([0-9.]+)', model_filename)
        if scale_match:
            scale_str = scale_match.group(1)
            if 'ceg' in model_filename:
                config_name = f'ceg_scale_{scale_str}_config.yaml'
            else:
                config_name = f'gm_scale_{scale_str}_config.yaml'
            config_path = os.path.join(model_dir, config_name)
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        train_cfg = yaml.safe_load(f)
                    is_fallback_cfg = True
                    print(f"[ Verification ] Loaded training config from fallback {config_path}")
                except Exception as e:
                    print(f"Warning: Failed to load fallback training config {config_path}: {e}")

    # 2. Try to query WandB logs as a secondary source (or primary if local file missing/fallback)
    try:
        import wandb
        project = None
        entity = None
        
        # Load synthetic_config.yaml to get entity and project
        project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        syn_cfg_path = os.path.join(project_dir, "guided_flow", "train", "synthetic_config.yaml")
        if os.path.exists(syn_cfg_path):
            with open(syn_cfg_path, "r") as sf:
                syn_cfg = yaml.safe_load(sf)
                project = syn_cfg.get("wandb", {}).get("project")
                entity = syn_cfg.get("wandb", {}).get("entity")

        if project:
            full_project_path = f"{entity}/{project}" if entity else project
            api = wandb.Api()
            # Find the run matching the file's timestamp or name parts
            timestamp_match = re.search(r'_(\d{8}_\d{6})\.pth', model_filename)
            if timestamp_match:
                ts = timestamp_match.group(1)
                runs = api.runs(path=full_project_path, filters={"name": {"$regex": ts}})
                
                # If no strict match on timestamp, search for the closest run by timestamp on the same day
                if not runs:
                    all_runs = api.runs(path=full_project_path)
                    import datetime
                    try:
                        checkpoint_dt = datetime.datetime.strptime(ts, "%Y%m%d_%H%M%S")
                    except Exception:
                        checkpoint_dt = None
                        
                    if checkpoint_dt:
                        best_run = None
                        min_time_diff = float('inf')
                        
                        # Find model type from filename
                        model_type = None
                        if 'ceg' in model_filename:
                            model_type = 'ceg'
                        else:
                            model_type_match = re.search(r'guidance_matching_(z|g|g2|g3|rwft)_scale_', model_filename)
                            if model_type_match:
                                model_type = model_type_match.group(1)
                                
                        for run in all_runs:
                            # Extract model type from run name
                            run_model_match = re.search(r'gm_(z|g|g2|g3|rwft)_scale_|ceg_scale_', run.name)
                            run_model_type = None
                            if run_model_match:
                                run_model_type = run_model_match.group(1) if run_model_match.group(1) else 'ceg'
                                
                            if model_type != run_model_type:
                                continue
                                
                            # Extract timestamp from run name
                            run_ts_match = re.search(r'_(\d{8}_\d{6})$', run.name)
                            if run_ts_match:
                                try:
                                    run_dt = datetime.datetime.strptime(run_ts_match.group(1), "%Y%m%d_%H%M%S")
                                    diff = abs((checkpoint_dt - run_dt).total_seconds())
                                    # The correct run starts within the same training session (usually under 2 hours)
                                    if diff < 7200 and diff < min_time_diff:
                                        min_time_diff = diff
                                        best_run = run
                                except Exception:
                                    pass
                        if best_run:
                            runs = [best_run]

                if runs:
                    run = runs[0]
                    print(f"[ Verification ] Found corresponding WandB run: {run.name}")
                    # Use WandB config if we don't have a strict 1-to-1 config
                    if not train_cfg or is_fallback_cfg:
                        wb_config = {}
                        for k, v in run.config.items():
                            wb_config[k] = v
                        train_cfg = wb_config
                        is_fallback_cfg = False
                        print("[ Verification ] Retrieved training config from WandB logs.")
    except Exception:
        pass

    if not train_cfg:
        print(f"Warning: Could not find any training logs (local or WandB) to verify energy function compatibility.")
        return

    # 3. Perform the actual compatibility validation checks
    # Check scale
    train_scale = float(train_cfg.get('scale', train_cfg.get('g_scale', 1.0)))
    if current_scale is not None and abs(train_scale - float(current_scale)) > 1e-5:
        raise ValueError(
            f"Scale mismatch for model at '{model_path}'! The loaded model was trained with scale {train_scale}, "
            f"but the notebook/inference is using scale {current_scale}."
        )

    # Check energy function name
    train_energy_fn = train_cfg.get('energy_fn', 'default')
    current_energy_fn = distribution.energy_fn_name

    resolved_train_energy_fn = train_energy_fn
    if train_energy_fn == 'default':
        resolved_train_energy_fn = distribution.get_default_energy_fn_name()

    resolved_current_energy_fn = current_energy_fn
    if current_energy_fn == 'default':
        resolved_current_energy_fn = distribution.get_default_energy_fn_name()

    if resolved_train_energy_fn != resolved_current_energy_fn:
        raise ValueError(
            f"Energy function mismatch for model at '{model_path}'! The loaded model was trained with energy function '{resolved_train_energy_fn}', "
            f"but the distribution is configured with '{resolved_current_energy_fn}'."
        )

    # Check energy function kwargs
    train_energy_kwargs_str = train_cfg.get('energy_kwargs', '{}')
    if isinstance(train_energy_kwargs_str, dict):
        train_energy_kwargs = train_energy_kwargs_str
    else:
        try:
            train_energy_kwargs = json.loads(train_energy_kwargs_str)
        except Exception:
            train_energy_kwargs = {}

    # Reconstruct energy_kwargs from WandB individual keys if energy_kwargs is missing or empty
    if not train_energy_kwargs:
        train_energy_kwargs = {}
        for k, v in train_cfg.items():
            if k.startswith('energy_') and k != 'energy_fn' and k != 'energy_kwargs':
                param_name = k[7:]
                train_energy_kwargs[param_name] = v

    # Resolve default arguments of the active energy function callable dynamically
    import inspect
    resolved_current_kwargs = {}
    try:
        energy_fn_callable = getattr(distribution, resolved_current_energy_fn)
        sig = inspect.signature(energy_fn_callable)
        for name, param in sig.parameters.items():
            if name in ['self', 'x', 'x1']:
                continue
            if param.default is not inspect.Parameter.empty:
                resolved_current_kwargs[name] = param.default
    except Exception:
        pass

    # Override defaults with explicitly set kwargs
    resolved_current_kwargs.update(distribution.energy_kwargs)

    # Compare keys and values
    for k, v in train_energy_kwargs.items():
        if k not in resolved_current_kwargs:
            raise ValueError(
                f"Energy hyperparameter mismatch for model at '{model_path}'! Key '{k}' was defined during training as {v}, "
                f"but is not applicable or missing in current distribution config."
            )
        if str(v) != str(resolved_current_kwargs[k]):
            raise ValueError(
                f"Energy hyperparameter mismatch for model at '{model_path}'! Parameter '{k}' was trained as {v}, "
                f"but is currently configured/resolved as {resolved_current_kwargs[k]}."
            )
    
    print("[ Verification ] Energy function and hyperparameter compatibility successfully verified!")

