import os
import torch
from guided_flow.config.guidance_training import GuidanceTrainingConfig
from guided_flow.distributions.base import get_distribution
import tyro
from matplotlib import pyplot as plt

from guided_flow.backbone.mlp import MLP, EnergyMLP
from guided_flow.flow.conditional_flow_matching import get_cfm
from guided_flow.utils.misc import deterministic, save_config, setup_device, get_device
import tqdm
import yaml
import datetime
import glob
from pathlib import Path

try:
    import wandb
    if not hasattr(wandb, "init"):
        wandb = None
except ImportError:
    wandb = None


def setup_wandb_run(
    model_type, 
    g_scale, 
    x0_dist, 
    x1_dist_obj, 
    cfm, 
    width, 
    sigma, 
    device, 
    batch_size, 
    lr, 
    num_steps,
    model_z_path=None,
    model_v_path=None
):
    config_path = Path(__file__).resolve().parent / "synthetic_config.yaml"
    wandb_config = None
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                synthetic_config = yaml.safe_load(f)
            wandb_config = synthetic_config.get("wandb", {})
        except Exception as e:
            print(f"Warning: Failed to load synthetic_config.yaml: {e}")

    if wandb is not None and wandb_config and wandb_config.get("project"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        x1_dist_name = str(x1_dist_obj)
        run_name = f"gm_{model_type}_scale_{g_scale}_{x0_dist}_{x1_dist_name}_{timestamp}"
        base_group = wandb_config.get("group")
        group_name = f"{base_group}_gm_{model_type}_scale_{g_scale}_{x0_dist}_{x1_dist_name}" if base_group else f"gm_{model_type}_scale_{g_scale}_{x0_dist}_{x1_dist_name}"
        
        # Build WandB config dynamically, including only relevant parameters
        config = {
            "model_name": f"guidance_matching_{model_type}",
            "g_scale": g_scale,
            "cfm": cfm,
            "x0_dist": x0_dist,
            "x1_dist": x1_dist_name,
            "width": width,
            "sigma": sigma,
            "device": str(device),
            "batch_size": batch_size,
            "lr": lr,
            "num_steps": num_steps,
            "energy_fn": x1_dist_obj.energy_fn_name,
        }
        for k, v in x1_dist_obj.energy_kwargs.items():
            config[f"energy_{k}"] = v

        if model_type in ["g", "g2", "g3", "rwft"]:
            if model_z_path is not None:
                config["model_z_path"] = str(model_z_path)
        if model_type in ["g2", "g3", "rwft"]:
            if model_v_path is not None:
                config["model_v_path"] = str(model_v_path)

        wandb.init(
            entity=wandb_config.get("entity"),
            project=wandb_config.get("project"),
            group=group_name,
            name=run_name,
            config=config,
            reinit="finish_previous"
        )
        return True
    return False


def train_z(
        x0_sampler,
        x1_sampler,
        # guidance related
        J,
        scale=1.0, 
        # training related
        cfm = 'cfm', 
        width = 64,
        sigma = 0.0,
        device=None,
        batch_size=1024,
        lr=1e-4,
        num_steps=20000,
        x0_dist='gaussian',
        x1_dist_obj=None
):
    if device is None:
        device = get_device()

    use_wandb = setup_wandb_run("z", scale, x0_dist, x1_dist_obj, cfm, width, sigma, device, batch_size, lr, num_steps)

    # Definition
    model = MLP(dim=2, out_dim=1, w=width, time_varying=True, exp_final=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    FM = get_cfm(cfm, sigma)

    with tqdm.tqdm(range(num_steps)) as pbar:
        for k in pbar:
            optimizer.zero_grad()

            x0 = x0_sampler(batch_size=batch_size, device=device)
            x1 = x1_sampler(batch_size=batch_size, device=device)

            t = torch.rand(x0.shape[0]).to(device)
            if cfm == 'ot_cfm':
                t, xt, ut, _, x1 = FM.guided_sample_location_and_conditional_flow(x0, x1, y0=None, y1=x1, t=t) # NOTE: we need correct x1!
            else:
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t)

            loss = torch.nn.functional.mse_loss(
                model(torch.cat([xt, t[:, None]], dim=-1)).squeeze(-1),
                torch.exp(-scale * J(x1))
            )

            loss.backward()
            optimizer.step()

            # Update progress bar with loss
            pbar.set_postfix(loss=loss.item())

            if use_wandb:
                wandb.log({"loss": loss.item()}, step=k)

    if use_wandb:
        wandb.finish()

    return model


def train_g(
        x0_sampler,
        x1_sampler,
        # guidance related
        model_z,
        J,
        scale=1.0, 
        # training related
        cfm = 'cfm', 
        width = 64,
        sigma = 0.0,
        device=None,
        batch_size=1024,
        lr=1e-4,
        num_steps=20000,
        x0_dist='gaussian',
        x1_dist_obj=None,
        model_z_path=None
):
    if device is None:
        device = get_device()

    use_wandb = setup_wandb_run("g", scale, x0_dist, x1_dist_obj, cfm, width, sigma, device, batch_size, lr, num_steps, model_z_path=model_z_path)

    # Model definition
    model = MLP(dim=2, out_dim=2, w=width, time_varying=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    FM = get_cfm(cfm, sigma)
    
    with tqdm.tqdm(range(num_steps)) as pbar:
        for k in pbar:
            optimizer.zero_grad()

            x0 = x0_sampler(batch_size=batch_size, device=device)
            x1 = x1_sampler(batch_size=batch_size, device=device)

            t = torch.rand(x0.shape[0]).to(device)
            if cfm == 'ot_cfm':
                t, xt, ut, _, x1 = FM.guided_sample_location_and_conditional_flow(x0, x1, y0=None, y1=x1, t=t) # NOTE: we need correct x1!
            else:
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t)

            # train guidance g
            z = model_z(torch.cat([xt, t[:, None]], dim=-1))
            g = model(torch.cat([xt, t[:, None]], dim=-1))
            loss = torch.nn.functional.mse_loss(
                g, # (B, 2)
                ((torch.exp(-scale * J(x1)).unsqueeze(-1) / (z.abs() + 1e-8)) - 1) * ut  # (B, 2)
            )

            loss.backward()
            optimizer.step()

            # Update progress bar with loss
            pbar.set_postfix(loss=loss.item())

            if use_wandb:
                wandb.log({"loss": loss.item()}, step=k)

    if use_wandb:
        wandb.finish()

    return model


def train_g2(
        x0_sampler,
        x1_sampler,
        # guidance related
        model_v,
        model_z,
        J,
        scale=1.0, 
        # training related
        cfm = 'cfm', 
        width = 64,
        sigma = 0.0,
        device=None,
        batch_size=1024,
        lr=1e-4,
        num_steps=20000,
        x0_dist='gaussian',
        x1_dist_obj=None,
        model_z_path=None,
        model_v_path=None
):
    if device is None:
        device = get_device()

    use_wandb = setup_wandb_run(
        "g2", scale, x0_dist, x1_dist_obj, cfm, width, sigma, device, batch_size, lr, num_steps,
        model_z_path=model_z_path, model_v_path=model_v_path
    )

    # Model definition
    model = MLP(dim=2, out_dim=2, w=width, time_varying=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    FM = get_cfm(cfm, sigma)
    
    with tqdm.tqdm(range(num_steps)) as pbar:
        for k in pbar:
            optimizer.zero_grad()

            x0 = x0_sampler(batch_size=batch_size, device=device)
            x1 = x1_sampler(batch_size=batch_size, device=device)

            t = torch.rand(x0.shape[0]).to(device)
            if cfm == 'ot_cfm':
                t, xt, ut, _, x1 = FM.guided_sample_location_and_conditional_flow(x0, x1, y0=None, y1=x1, t=t) # NOTE: we need correct x1!
            else:
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t)

            # train guidance g
            z = model_z(torch.cat([xt, t[:, None]], dim=-1))
            v = model_v(torch.cat([xt, t[:, None]], dim=-1))
            g = model(torch.cat([xt, t[:, None]], dim=-1))
            loss = torch.nn.functional.mse_loss(
                g, # (B, 2)
                (torch.exp(-scale * J(x1)).unsqueeze(-1) / (z.abs() + 1e-8)) * ut  # (B, 2)
                - v
            )

            loss.backward()
            optimizer.step()

            # Update progress bar with loss
            pbar.set_postfix(loss=loss.item())

            if use_wandb:
                wandb.log({"loss": loss.item()}, step=k)

    if use_wandb:
        wandb.finish()

    return model


def train_g3(
        x0_sampler,
        x1_sampler,
        # guidance related
        model_v,
        model_z,
        J,
        scale=1.0, 
        # training related
        cfm = 'cfm', 
        width = 64,
        sigma = 0.0,
        device=None,
        batch_size=1024,
        lr=1e-4,
        num_steps=20000,
        x0_dist='gaussian',
        x1_dist_obj=None,
        model_z_path=None,
        model_v_path=None
):
    if device is None:
        device = get_device()

    use_wandb = setup_wandb_run(
        "g3", scale, x0_dist, x1_dist_obj, cfm, width, sigma, device, batch_size, lr, num_steps,
        model_z_path=model_z_path, model_v_path=model_v_path
    )

    # Model definition
    model = MLP(dim=2, out_dim=2, w=width, time_varying=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    FM = get_cfm(cfm, sigma)
    
    with tqdm.tqdm(range(num_steps)) as pbar:
        for k in pbar:
            optimizer.zero_grad()

            x0 = x0_sampler(batch_size=batch_size, device=device)
            x1 = x1_sampler(batch_size=batch_size, device=device)

            t = torch.rand(x0.shape[0]).to(device)
            if cfm == 'ot_cfm':
                t, xt, ut, _, x1 = FM.guided_sample_location_and_conditional_flow(x0, x1, y0=None, y1=x1, t=t) # NOTE: we need correct x1!
            else:
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t)

            # train guidance g
            z = model_z(torch.cat([xt, t[:, None]], dim=-1))
            v = model_v(torch.cat([xt, t[:, None]], dim=-1))
            g = model(torch.cat([xt, t[:, None]], dim=-1))
            loss = (
                torch.square(
                    g + v - ut  # (B, 2)
                ) * (torch.exp(-scale * J(x1).unsqueeze(-1)) / (z.abs() + 1e-8))
            ).mean()

            loss.backward()
            optimizer.step()

            # Update progress bar with loss
            pbar.set_postfix(loss=loss.item())

            if use_wandb:
                wandb.log({"loss": loss.item()}, step=k)

    if use_wandb:
        wandb.finish()

    return model


def train_rwft(
        x0_sampler,
        x1_sampler,
        # guidance related
        model_v,
        model_z,
        J,
        scale=1.0, 
        # training related
        cfm = 'cfm', 
        width = 64,
        sigma = 0.0,
        device=None,
        batch_size=1024,
        lr=1e-4,
        num_steps=20000,
        x0_dist='gaussian',
        x1_dist_obj=None,
        model_z_path=None,
        model_v_path=None
):
    if device is None:
        device = get_device()

    use_wandb = setup_wandb_run(
        "rwft", scale, x0_dist, x1_dist_obj, cfm, width, sigma, device, batch_size, lr, num_steps,
        model_z_path=model_z_path, model_v_path=model_v_path
    )

    # Model definition
    model = MLP(dim=2, out_dim=2, w=width, time_varying=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    FM = get_cfm(cfm, sigma)
    
    with tqdm.tqdm(range(num_steps)) as pbar:
        for k in pbar:
            optimizer.zero_grad()

            x0 = x0_sampler(batch_size=batch_size, device=device)
            x1 = x1_sampler(batch_size=batch_size, device=device)

            t = torch.rand(x0.shape[0]).to(device)
            if cfm == 'ot_cfm':
                t, xt, ut, _, x1 = FM.guided_sample_location_and_conditional_flow(x0, x1, y0=None, y1=x1, t=t) # NOTE: we need correct x1!
            else:
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t)

            # train guidance g
            z = model_z(torch.cat([xt, t[:, None]], dim=-1))
            v = model_v(torch.cat([xt, t[:, None]], dim=-1))
            g = model(torch.cat([xt, t[:, None]], dim=-1))
            loss = (
                torch.square(
                    g + v - ut  # (B, 2)
                ) * (torch.exp(-scale * J(x1).unsqueeze(-1)))
            ).mean()

            loss.backward()
            optimizer.step()

            # Update progress bar with loss
            pbar.set_postfix(loss=loss.item())

            if use_wandb:
                wandb.log({"loss": loss.item()}, step=k)

    if use_wandb:
        wandb.finish()

    return model


if __name__ == "__main__":
    cfg = tyro.cli(GuidanceTrainingConfig)

    deterministic(cfg.seed)
    setup_device(cfg)

    # save the config
    log_subfolder = os.path.join(
        "logs", f"{cfg.x0_dist}-{cfg.x1_dist}", f"{cfg.cfm}_{cfg.x0_dist}_{cfg.x1_dist}"
    )
    save_config(cfg, log_subfolder, config_name=f'gm_scale_{cfg.scale}_config.yaml')

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build distributions
    x0_dist_obj = get_distribution(cfg.x0_dist)
    
    import json
    try:
        energy_kwargs_dict = json.loads(cfg.energy_kwargs)
    except Exception as e:
        raise ValueError(f"Failed to parse energy_kwargs as JSON: {e}")
        
    x1_dist_obj = get_distribution(
        cfg.x1_dist, 
        energy_fn=None if cfg.energy_fn == 'default' else cfg.energy_fn, 
        **energy_kwargs_dict
    )

    # train z
    if 'z' in cfg.which_model.split('-'):
        model_z = train_z(
            x0_dist_obj.sample, 
            x1_dist_obj.sample, 
            x1_dist_obj.get_J, 
            cfg.scale, 
            cfg.cfm, 
            cfg.width, 
            cfg.sigma, 
            cfg.device, 
            cfg.batch_size, 
            cfg.lr, 
            cfg.num_steps,
            x0_dist=cfg.x0_dist,
            x1_dist_obj=x1_dist_obj
        )

        model_path = os.path.join(log_subfolder, f"guidance_matching_z_scale_{cfg.scale}_{cfg.x0_dist}_{cfg.x1_dist}_{timestamp}.pth")
        torch.save(model_z.state_dict(), model_path)
        save_config(cfg, log_subfolder, config_name=os.path.basename(model_path).replace(".pth", ".yaml"))

    # load model_z in case not trained from scratch
    model_z = MLP(dim=2, out_dim=1, w=cfg.width, time_varying=True, exp_final=True).to(cfg.device)
    model_z_search_pattern = os.path.join(log_subfolder, f"guidance_matching_z_scale_{cfg.scale}_{cfg.x0_dist}_{cfg.x1_dist}*.pth")
    model_z_files = glob.glob(model_z_search_pattern)

    if not model_z_files:
        raise FileNotFoundError(f"No model_z found matching: {model_z_search_pattern}.")
    
    # Sort alphabetically to get the newest model path
    latest_model_z_path = sorted(model_z_files)[-1]
    print(f"Loading model_z from: {latest_model_z_path}")
    model_z.load_state_dict(torch.load(latest_model_z_path, map_location=cfg.device, weights_only=True))

    # train g
    if 'g' in cfg.which_model.split('-'):
        model_g = train_g(
            x0_dist_obj.sample, 
            x1_dist_obj.sample, 
            model_z, 
            x1_dist_obj.get_J, 
            cfg.scale, 
            cfg.cfm, 
            cfg.width, 
            cfg.sigma, 
            cfg.device, 
            cfg.batch_size, 
            cfg.lr, 
            cfg.num_steps,
            x0_dist=cfg.x0_dist,
            x1_dist_obj=x1_dist_obj,
            model_z_path=latest_model_z_path
        )

        model_path = os.path.join(log_subfolder, f"guidance_matching_g_scale_{cfg.scale}_{cfg.x0_dist}_{cfg.x1_dist}_{timestamp}.pth")
        torch.save(model_g.state_dict(), model_path)
        save_config(cfg, log_subfolder, config_name=os.path.basename(model_path).replace(".pth", ".yaml"))


    # load model_v
    latest_cfm_path = None
    if 'g2' in cfg.which_model.split('-') or 'g3' in cfg.which_model.split('-') or 'rwft' in cfg.which_model.split('-'):
        # Initialize the base model architecture
        model_v = MLP(dim=2, w=cfg.width, time_varying=True).to(cfg.device)

        # 1. Search for the latest trained base model
        cfm_search_pattern = os.path.join(log_subfolder, f"{cfg.cfm}_{cfg.x0_dist}_{cfg.x1_dist}*.pth")
        cfm_files = glob.glob(cfm_search_pattern)

        if not cfm_files:
            raise FileNotFoundError(f"No model_v found matching: {cfm_search_pattern}. Please train the base CFM model first.")

        # Sort alphabetically to get the newest timestamp
        latest_cfm_path = sorted(cfm_files)[-1]
        print(f"Loading base CFM model from: {latest_cfm_path}")

        # 2. Load the model
        model_v.load_state_dict(torch.load(latest_cfm_path, map_location=cfg.device, weights_only=True))
        model_v.eval() # Good practice to set it to eval mode

    # train g2
    if 'g2' in cfg.which_model.split('-'):
        model_g2 = train_g2(
            x0_dist_obj.sample, 
            x1_dist_obj.sample, 
            model_v, 
            model_z, 
            x1_dist_obj.get_J, 
            cfg.scale, 
            cfg.cfm, 
            cfg.width, 
            cfg.sigma, 
            cfg.device, 
            cfg.batch_size, 
            cfg.lr, 
            cfg.num_steps,
            x0_dist=cfg.x0_dist,
            x1_dist_obj=x1_dist_obj,
            model_z_path=latest_model_z_path,
            model_v_path=latest_cfm_path
        )

        model_path = os.path.join(log_subfolder, f"guidance_matching_g2_scale_{cfg.scale}_{cfg.x0_dist}_{cfg.x1_dist}_{timestamp}.pth")
        torch.save(model_g2.state_dict(), model_path)
        save_config(cfg, log_subfolder, config_name=os.path.basename(model_path).replace(".pth", ".yaml"))

    # train g3
    if 'g3' in cfg.which_model.split('-'):
        model_g3 = train_g3(
            x0_dist_obj.sample, 
            x1_dist_obj.sample, 
            model_v, 
            model_z, 
            x1_dist_obj.get_J, 
            cfg.scale, 
            cfg.cfm, 
            cfg.width, 
            cfg.sigma, 
            cfg.device, 
            cfg.batch_size, 
            cfg.lr, 
            cfg.num_steps,
            x0_dist=cfg.x0_dist,
            x1_dist_obj=x1_dist_obj,
            model_z_path=latest_model_z_path,
            model_v_path=latest_cfm_path
        )

        model_path = os.path.join(log_subfolder, f"guidance_matching_g3_scale_{cfg.scale}_{cfg.x0_dist}_{cfg.x1_dist}_{timestamp}.pth")
        torch.save(model_g3.state_dict(), model_path)
        save_config(cfg, log_subfolder, config_name=os.path.basename(model_path).replace(".pth", ".yaml"))

    # train rwft
    if 'rwft' in cfg.which_model.split('-'):
        model_rwft = train_rwft(
            x0_dist_obj.sample, 
            x1_dist_obj.sample, 
            model_v, 
            model_z, 
            x1_dist_obj.get_J, 
            cfg.scale, 
            cfg.cfm, 
            cfg.width, 
            cfg.sigma, 
            cfg.device, 
            cfg.batch_size, 
            cfg.lr, 
            cfg.num_steps,
            x0_dist=cfg.x0_dist,
            x1_dist_obj=x1_dist_obj,
            model_z_path=latest_model_z_path,
            model_v_path=latest_cfm_path
        )

        model_path = os.path.join(log_subfolder, f"guidance_matching_rwft_scale_{cfg.scale}_{cfg.x0_dist}_{cfg.x1_dist}_{timestamp}.pth")
        torch.save(model_rwft.state_dict(), model_path)
        save_config(cfg, log_subfolder, config_name=os.path.basename(model_path).replace(".pth", ".yaml"))
