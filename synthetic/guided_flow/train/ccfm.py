import os
import argparse
import copy
import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from pathlib import Path
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher

# Import your custom modules (assuming standard project structure)
from guided_flow.backbone.mlp import CondMLP
from guided_flow.distributions.base import get_distribution
from guided_flow.utils.misc import deterministic

try:
    import wandb
except ImportError:
    wandb = None


def train_cc_cfm_ot_early_stop(source_dist_name, target_dist_name, device="cuda", epochs=15000, batch_size=512, w=256):
    print(f"Training OT-CC-CFM Baseline: {source_dist_name} -> {target_dist_name}")
    
    # Load wandb config
    config_path = Path(__file__).resolve().parent / "synthetic_config.yaml"
    wandb_config = None
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                synthetic_config = yaml.safe_load(f)
            wandb_config = synthetic_config.get("wandb", {})
        except Exception as e:
            print(f"Warning: Failed to load synthetic_config.yaml: {e}")

    use_wandb = False
    if wandb is not None and wandb_config and wandb_config.get("project"):
        use_wandb = True

    if use_wandb:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"ccfm_{source_dist_name}_{target_dist_name}_{timestamp}"
        base_group = wandb_config.get("group")
        group_name = f"{base_group}_ccfm_{source_dist_name}_{target_dist_name}" if base_group else f"ccfm_{source_dist_name}_{target_dist_name}"
        wandb.init(
            entity=wandb_config.get("entity"),
            project=wandb_config.get("project"),
            group=group_name,
            name=run_name,
            config={
                "model_name": "ccfm",
                "source_dist": source_dist_name,
                "target_dist": target_dist_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "w": w,
                "device": device
            },
            reinit=True
        )

    model_cc = CondMLP(dim=2, w=w, num_classes=8).to(device)
    optimizer = optim.Adam(model_cc.parameters(), lr=1e-3)
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
    
    x0_dist = get_distribution(source_dist_name)
    x1_dist = get_distribution(target_dist_name)

    tracked_steps = []
    tracked_losses = []
    best_loss = float('inf')
    best_model_state = None
    increase_count = 0

    model_cc.train()
    for step in range(epochs):
        optimizer.zero_grad()
        
        x0 = x0_dist.sample(batch_size).to(device)
        x1 = x1_dist.sample(batch_size).to(device)
        
        # Auto-label based on spatial angle (specific to 8-Gaussian geometry)
        angles = torch.atan2(x1[:, 1], x1[:, 0])
        y = torch.round((angles + torch.pi) / (torch.pi / 4)) % 8
        y = y.long().to(device)
        
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        xt_time = torch.cat([xt, t[:, None]], dim=-1)

        # give conditional class label y as model input
        vt = model_cc(xt_time, y)
        
        loss = F.mse_loss(vt, ut)
        loss.backward()
        optimizer.step()

        if use_wandb:
            wandb.log({"loss": loss.item()}, step=step)
        
        if step % 500 == 0:
            current_loss = loss.item()
            tracked_steps.append(step)
            tracked_losses.append(current_loss)
            print(f"Step {step:05d} | Loss: {current_loss:.4f}")
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_model_state = copy.deepcopy(model_cc.state_dict()) 
                increase_count = 0  
            else:
                increase_count += 1
                print(f"  -> Loss increased! Warning count: {increase_count}/2")
                
            if increase_count >= 2:
                print(f"Early stopping triggered at step {step}!")
                break
                
    print(f"Training finished. Best Loss: {best_loss:.4f}")
    model_cc.load_state_dict(best_model_state)

    if use_wandb:
        wandb.finish()

    return model_cc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", "--x0-dist", dest="source", type=str, default="uniform", help="Source distribution")
    parser.add_argument("--target", "--x1-dist", dest="target", type=str, default="8gaussian", help="Target distribution")
    parser.add_argument("--epochs", type=int, default=15000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str,
        default="cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    args = parser.parse_args()

    deterministic(0)
    
    # 1. Train the model
    model = train_cc_cfm_ot_early_stop(
        args.source, args.target, 
        device=args.device, epochs=args.epochs, batch_size=args.batch_size
    )

    # 2. Dynamically resolve the absolute path to the project's root 'logs' directory
    # (Assuming this script is located in `scripts/` or `guided_flow/train/`)
    current_file_path = Path(__file__).resolve()
    
    # Adjust the parent depending on how deep this file is. 
    # E.g., if it's in `scripts/train.py`, parents[1] is the project root.
    project_root = current_file_path.parents[2] 
    
    # Create the target directory: logs/{source}-{target}/cc_cfm_{source}_{target}/
    log_dir = project_root / "logs" / f"{args.source}-{args.target}" / f"cc_cfm_{args.source}_{args.target}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Save the weights
    save_path = log_dir / f"cc_cfm_{args.source}_{args.target}_best.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Successfully saved weights to: {save_path}")
