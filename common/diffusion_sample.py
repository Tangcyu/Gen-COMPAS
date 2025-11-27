import os
import time
import yaml
import torch
import numpy as np
import mdtraj as md
from tqdm import tqdm
from utils.model import DiffusionModel
from utils.diffusion import Diffusion
from utils.logger import get_logger

logger = get_logger(__name__)

def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    except yaml.YAMLError as e:
        raise RuntimeError(f"Error parsing YAML file: {e}")


def setup_device(device_str: str | None) -> torch.device:
    """Initialize device (CUDA if available)."""
    device = torch.device(device_str if device_str else ('cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")
    return device


def setup_model_and_diffusion(config: dict, device: torch.device):
    """Load model, diffusion process, and normalization constants."""
    checkpoint_path = config['inference']['checkpoint']
    psf_path = config['data']['psf_path']
    save_dir = os.path.dirname(checkpoint_path)

    # Load normalization constants
    coord_mean = torch.load(os.path.join(save_dir, 'coord_mean.pt'), map_location=device).view(1, 1, 3)
    coord_std = torch.load(os.path.join(save_dir, 'coord_std.pt'), map_location=device).view(1, 1, 3)
    logger.info("Loaded normalization constants (mean/std).")

    # Load topology
    topology = md.load_psf(psf_path)
    num_atoms = topology.n_atoms
    atom_names = [atom.name for atom in topology.atoms]
    logger.info(f"Topology loaded: {num_atoms} atoms.")

    # Initialize model
    model_cfg = config['model']
    model = DiffusionModel(
        num_atoms=num_atoms,
        topology=topology,
        atom_types=atom_names,
        node_feature_dim=model_cfg['node_feature_dim'],
        time_embedding_dim=model_cfg['time_embedding_dim'],
        hidden_dim=model_cfg['hidden_dim'],
        num_schnet_layers=model_cfg['num_schnet_layers'],
        num_gat_layers=model_cfg['num_gat_layers'],
        residue_attn_heads=model_cfg['residue_attn_heads'],
        k_neighbors=model_cfg['k_neighbors']
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    logger.info(f"Loaded model weights from checkpoint: {checkpoint_path}")

    # Initialize diffusion
    diff_cfg = config['diffusion']
    diffusion = Diffusion(
        timesteps=diff_cfg['timesteps'],
        beta_schedule=diff_cfg['beta_schedule'],
        device=device
    )

    return model, diffusion, coord_mean, coord_std, topology


@torch.no_grad()
def generate_samples(model, diffusion, num_samples, num_atoms, device, noise_scale, sample_batch):
    """Generate structures using diffusion model in batches."""
    all_coords = []
    num_batches = (num_samples + sample_batch - 1) // sample_batch
    logger.info(f"Generating {num_samples} samples in {num_batches} batches of {sample_batch}...")

    for batch_idx in range(num_batches):
        current_batch = min(sample_batch, num_samples - batch_idx * sample_batch)
        logger.info(f"Processing batch {batch_idx + 1}/{num_batches} (size={current_batch})")

        x_t = torch.randn(current_batch, num_atoms, 3, device=device)

        for t_val in tqdm(
            reversed(range(diffusion.timesteps)),
            desc=f"Batch {batch_idx + 1}/{num_batches}",
            total=diffusion.timesteps,
            leave=False
        ):
            t = torch.full((current_batch,), t_val, device=device, dtype=torch.long)
            pred_x0 = model(x_t, t)
            x_t = diffusion.p_sample(pred_x0, x_t, t, noise_scale=noise_scale)

        all_coords.append(x_t.cpu())
        del x_t, pred_x0
        torch.cuda.empty_cache()

    coords = torch.cat(all_coords, dim=0)
    logger.info(f"Sampling completed: {len(coords)} structures generated.")
    return coords


def unnormalize_coords(coords: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Undo normalization."""
    return coords * std.to(coords.device) + mean.to(coords.device)


def save_to_pdb(coords: np.ndarray, topology, output_path: str):
    """Save coordinates as PDB trajectory."""
    traj = md.Trajectory(xyz=coords, topology=topology)
    traj.save_pdb(output_path)
    logger.info(f"Saved {len(coords)} structures to {output_path}")


# =========================================================
# Main Function
# =========================================================

def run_diffusion_inference(config: dict):
    """Run Diffusion Model inference."""
    start_time = time.time()
    # config = load_config(config_path)

    # Assign config values
    device = setup_device(config.get('device'))
    inference_cfg = config['inference']

    output_path = inference_cfg['output'] + ".pdb"
    num_samples = inference_cfg['num_samples']
    noise_scale = inference_cfg['noise_scale']
    sample_batch = inference_cfg.get('sample_batch', 10)

    # === Load model and diffusion ===
    model, diffusion, mean, std, topology = setup_model_and_diffusion(config, device)
    num_atoms = topology.n_atoms

    # === Sampling ===
    coords = generate_samples(model, diffusion, num_samples, num_atoms, device, noise_scale, sample_batch)
    coords_unnorm = unnormalize_coords(coords, mean, std).cpu().numpy()

    # === Save results ===
    save_to_pdb(coords_unnorm, topology, output_path)

    logger.info(f"Inference completed in {(time.time() - start_time)/60:.2f} minutes.")


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate protein structures Diffusion Model")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    run_diffusion_inference(args.config['Generative'])