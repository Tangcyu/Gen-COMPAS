from __future__ import annotations

import torch


def weighted_jab(q0: torch.Tensor, qt: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights / torch.clamp(torch.sum(weights), min=1e-12)
    return torch.sum(weights * torch.square(q0 - qt))


def endpoint_restraint(
    q0: torch.Tensor,
    qt: torch.Tensor,
    ka0: torch.Tensor,
    kat: torch.Tensor,
    kb0: torch.Tensor,
    kbt: torch.Tensor,
    center0: torch.Tensor,
    centert: torch.Tensor,
) -> torch.Tensor:
    res_a = ka0 * torch.square(q0 - center0) + kat * torch.square(qt - centert)
    res_b = kb0 * torch.square(q0 - center0) + kbt * torch.square(qt - centert)
    return torch.mean(res_a + res_b)


def add_normalized_noise(z: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0.0:
        return z
    return z + torch.randn_like(z) * std


def gradient_l2_penalty(model, z: torch.Tensor, max_points: int = 0) -> torch.Tensor:
    if max_points and z.shape[0] > max_points:
        idx = torch.randperm(z.shape[0], device=z.device)[:max_points]
        z = z[idx]
    z_req = z.detach().clone().requires_grad_(True)
    q = model.forward_normalized(z_req)
    grad = torch.autograd.grad(q.sum(), z_req, create_graph=True)[0]
    return torch.mean(torch.square(grad))


def committor_loss(model, batch, config: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x0, xt, weights, ka0, kat, kb0, kbt, center0, centert = batch
    z0 = model.normalize_input(x0)
    zt = model.normalize_input(xt)
    z0_eval = add_normalized_noise(z0, float(config.get("input_noise_std", 0.0)))
    zt_eval = add_normalized_noise(zt, float(config.get("input_noise_std", 0.0)))
    q0 = model.forward_normalized(z0_eval)
    qt = model.forward_normalized(zt_eval)

    jab = weighted_jab(q0, qt, weights)
    restraint = endpoint_restraint(q0, qt, ka0, kat, kb0, kbt, center0, centert)
    total = jab + float(config.get("endpoint_scale", config.get("k_scale", 100.0))) * restraint

    grad_penalty = torch.zeros((), device=x0.device)
    grad_scale = float(config.get("gradient_l2_scale", 0.0))
    if grad_scale > 0.0:
        grad_penalty = gradient_l2_penalty(model, z0, int(config.get("gradient_l2_max_points", 4096)))
        total = total + grad_scale * grad_penalty

    return total, {
        "loss": total.detach(),
        "jab": jab.detach(),
        "endpoint": restraint.detach(),
        "grad_l2": grad_penalty.detach(),
    }
