# utils/diffusion.py
import torch
from typing import Tuple

def center_coords(coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Centers coordinates by subtracting their mean (center of mass).
    
    Args:
        coords: Shape (B, N, 3) or (N, 3)
    
    Returns:
        centered_coords: Same shape as input
        mean: Shape (B, 1, 3) or (1, 3)
    """
    mean = torch.mean(coords, dim=-2, keepdim=True)
    centered_coords = coords - mean
    return centered_coords, mean

def uncenter_coords(centered_coords: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """
    Adds back the center of mass to centered coordinates.
    Handles shape mismatches for broadcasting.
    """
    if center.ndim == 2 and centered_coords.ndim == 3:
         center = center.unsqueeze(1)
    elif center.ndim == 1 and centered_coords.ndim >= 2:
         num_leading_dims = centered_coords.ndim - 1
         center_shape = (1,) * num_leading_dims + (center.shape[0],)
         center = center.view(center_shape)

    return centered_coords + center


class Diffusion:
    """Implements variance scheduling and sampling"""
    
    def __init__(self, timesteps: int = 1000, beta_schedule: str = 'cosine', device: str = 'cuda'):
        self.timesteps = timesteps
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        if beta_schedule == 'cosine':
            self.betas = self.cosine_beta_schedule(timesteps, s=0.008).to(self.device, dtype=torch.float32)
        elif beta_schedule == 'linear':
            self.betas = self.linear_beta_schedule(timesteps).to(self.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        print(f"Initialized Diffusion with {timesteps} steps, schedule '{beta_schedule}', device '{self.device}'")
        self._precompute_terms()

    def cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine variance schedule - smoother than linear, better for generation."""
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        f_t = torch.cos(((t / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = f_t / f_t[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, min=1e-6, max=0.02)

    def linear_beta_schedule(self, timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
        """Linear variance schedule."""
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

    def _precompute_terms(self):
        """Precompute constants for efficient diffusion operations."""
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]])

        # Forward process: q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Reverse process posterior: q(x_{t-1} | x_t, x_0)
        variance_denom = (1.0 - self.alphas_cumprod)
        self.posterior_variance = torch.where(
            variance_denom > 1e-10,
             self.betas * (1.0 - self.alphas_cumprod_prev) / variance_denom,
             torch.zeros_like(self.betas)
        )
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))

        # Posterior mean coefficients
        self.posterior_mean_coef1 = torch.where(
            variance_denom > 1e-10,
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / variance_denom,
            torch.zeros_like(self.betas)
        )
        self.posterior_mean_coef2 = torch.where(
            variance_denom > 1e-10,
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / variance_denom,
            torch.ones_like(self.betas)
        )

        # For predicting x_0 from noise
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """Extract values at timestep indices and reshape for broadcasting."""
        batch_size = t.shape[0]
        out = a.gather(-1, t.to(a.device).long())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def add_noise(self, x_start: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: adds noise to clean coordinates.

        Args:
            x_start: Clean coordinates (can be uncentered) [B, N, 3]
            t: Timesteps for each batch element [B]

        Returns:
            x_noisy_centered: Noisy coordinates (centered) [B, N, 3]
            noise: Sampled Gaussian noise [B, N, 3]
            center: Original center of mass [B, 1, 3]
        """
        x_centered, center = center_coords(x_start)
        noise = torch.randn_like(x_centered, device=self.device)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_centered.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_centered.shape)

        x_noisy_centered = sqrt_alphas_cumprod_t * x_centered + sqrt_one_minus_alphas_cumprod_t * noise

        return x_noisy_centered, noise, center

    def predict_x0_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor) -> torch.Tensor:
        """
        Predict clean x_0 from noisy x_t and predicted noise.
        All inputs/outputs are in centered space.
        """
        assert x_t.ndim == 3 and x_t.shape[-1] == 3

        sqrt_recip_alphas_cumprod_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

        x0_pred_centered = sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise_pred
        return x0_pred_centered

    def p_sample(self, model_output_x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, noise_scale: float = 1.0) -> torch.Tensor:
        """
        Reverse diffusion step: sample x_{t-1} given x_t and predicted x_0.
        All coordinates are in centered space.

        Args:
            model_output_x0: Predicted clean coordinates (centered) [B, N, 3]
            x_t: Current noisy coordinates (centered) [B, N, 3]
            t: Current timestep [B]
            noise_scale: Noise injection scale (0 = deterministic)

        Returns:
            x_{t-1}: Denoised coordinates (centered) [B, N, 3]
        """
        assert model_output_x0.ndim == 3 and model_output_x0.shape[-1] == 3
        assert x_t.ndim == 3 and x_t.shape[-1] == 3
        x0_pred_centered = model_output_x0

        # Get posterior distribution parameters
        posterior_mean_coef1_t = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2_t = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_log_variance_t = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)

        # Compute posterior mean
        posterior_mean = posterior_mean_coef1_t * x0_pred_centered + posterior_mean_coef2_t * x_t

        # Sample noise (or use zeros for deterministic sampling)
        noise = torch.randn_like(x_t, device=self.device) if noise_scale > 0 else torch.zeros_like(x_t, device=self.device)

        # Don't add noise at t=0
        nonzero_mask = (t != 0).float().view(-1, *((1,) * (len(x_t.shape) - 1)))
        nonzero_mask = nonzero_mask.to(self.device)

        x_prev_centered = posterior_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance_t) * noise * noise_scale

        return x_prev_centered