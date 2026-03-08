# train.py
import os
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from tqdm import tqdm

# Project imports
from utils.data_loader import ProteinDataset
from utils.model import DiffusionModel
from utils.diffusion import Diffusion, center_coords
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------
# Configuration and setup utilities
# ---------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load a YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        raise RuntimeError(f"Error parsing YAML config: {e}")


def setup_device(device_str: str) -> torch.device:
    """Set up the CUDA or CPU device."""
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    return device


def setup_dataloader(data_cfg: dict, training_cfg: dict):
    """Initialize the dataset and data loader."""
    dataset = ProteinDataset(
        psf_path=data_cfg['psf_path'],
        dcd_path=data_cfg['dcd_path']
    )
    loader = DataLoader(
        dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=training_cfg.get('num_workers', 4),
        drop_last=True
    )
    logger.info(f"Dataset: {len(dataset)} samples, {dataset.num_atoms} atoms per sample.")
    return dataset, loader


def setup_model(model_cfg: dict, dataset, device: torch.device, init_ckpt: str = None):
    """Create and initialize the diffusion model."""
    model = DiffusionModel(
        num_atoms=dataset.num_atoms,
        topology=dataset.topology,
        atom_types=[atom.name for atom in dataset.topology.atoms],
        node_feature_dim=model_cfg['node_feature_dim'],
        time_embedding_dim=model_cfg['time_embedding_dim'],
        hidden_dim=model_cfg['hidden_dim'],
        num_schnet_layers=model_cfg['num_schnet_layers'],
        num_gat_layers=model_cfg['num_gat_layers'],
        residue_attn_heads=model_cfg['residue_attn_heads'],
        k_neighbors=model_cfg['k_neighbors']
    ).to(device)

    if init_ckpt:
        model.load_state_dict(torch.load(init_ckpt, map_location=device))
        logger.info(f"Loaded model weights from checkpoint: {init_ckpt}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params / 1e6:.2f} M")

    return model


def setup_optimizer_scheduler(model, training_cfg, steps_per_epoch):
    """Set up the optimizer and learning rate scheduler."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg['lr']),
        weight_decay=float(training_cfg['weight_decay'])
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(training_cfg['lr']),
        epochs=training_cfg['epochs'],
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1
    )
    return optimizer, scheduler


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------

def train_diffusion_model(config: dict):
    """Train the diffusion model."""
    start_time = time.time()

    # Set up directories and devices
    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    device = setup_device(config['device'])

    data_cfg = config['data']
    model_cfg = config['model']
    diffusion_cfg = config['diffusion']
    training_cfg = config['training']

    grad_clip_value = training_cfg.get('grad_clip', 0.0)
    save_interval = training_cfg.get('save_interval', 50)
    init_ckpt = config.get('init_checkpoint_path', None)

    # Save configuration for reproducibility
    config_path = os.path.join(save_dir, str(time.ctime())+'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    logger.info(f"Configuration saved to {config_path}")

    # Prepare data and normalization constants
    dataset, loader = setup_dataloader(data_cfg, training_cfg)
    coord_mean, coord_std = dataset.get_normalization_constants()
    torch.save(coord_mean.cpu(), os.path.join(save_dir, 'coord_mean.pt'))
    torch.save(coord_std.cpu(), os.path.join(save_dir, 'coord_std.pt'))
    logger.info("Saved normalization constants (mean/std).")

    # Initialize model and diffusion process
    model = setup_model(model_cfg, dataset, device, init_ckpt)
    diffusion = Diffusion(
        timesteps=diffusion_cfg['timesteps'],
        beta_schedule=diffusion_cfg['beta_schedule'],
        device=device
    )

    optimizer, scheduler = setup_optimizer_scheduler(model, training_cfg, len(loader))

    # Enable mixed precision if CUDA is available
    use_amp = device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    logger.info(f"Using AMP (mixed precision): {use_amp}")

    logger.info("Starting training...")
    best_loss = float('inf')

    for epoch in range(training_cfg['epochs']):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{training_cfg['epochs']}", leave=True)

        for step, batch in enumerate(progress):
            x0_uncentered = batch['coords'].to(device)
            t = torch.randint(0, diffusion_cfg['timesteps'], (x0_uncentered.size(0),), device=device)

            x_noisy_centered, _, _ = diffusion.add_noise(x0_uncentered, t)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                pred_x0_centered = grad_checkpoint(model, x_noisy_centered, t, use_reentrant=False)
                x0_centered, _ = center_coords(x0_uncentered)
                loss = F.mse_loss(pred_x0_centered, x0_centered)

            if torch.isnan(loss):
                logger.warning(f"NaN loss at Epoch {epoch+1}, Step {step} — skipping.")
                continue

            # Backpropagation and optimization
            scaler.scale(loss).backward()

            if grad_clip_value > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.3e}")

        avg_loss = epoch_loss / len(loader)
        logger.info(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.3e}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(save_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved ({avg_loss:.5f})")

        # Periodic checkpoint saving
        if (epoch + 1) % save_interval == 0:
            ckpt_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    # Save final model
    final_model_path = os.path.join(save_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training completed. Final model saved to {final_model_path}")
    logger.info(f"Total training time: {(time.time() - start_time)/3600:.2f} hours")


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the diffusion model")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    config = load_config(args.config)
    train_diffusion_model(config['Generative'])
