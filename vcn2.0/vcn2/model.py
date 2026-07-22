from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def activation_from_name(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name in ("identity", "linear", "none"):
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


class InputNormalizer(nn.Module):
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, clip: float = 0.0):
        super().__init__()
        self.register_buffer("loc", loc.detach().clone().float())
        self.register_buffer("scale", scale.detach().clone().float())
        self.clip = float(clip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.loc) / self.scale
        if self.clip > 0.0:
            z = torch.clamp(z, -self.clip, self.clip)
        return z

    @torch.jit.export
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    @torch.jit.export
    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.scale + self.loc


class CommittorNet(nn.Module):
    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        hidden_layers: list[int],
        activation: str = "elu",
        input_clip: float = 0.0,
        input_dropout: float = 0.0,
    ):
        super().__init__()
        self.normalizer = InputNormalizer(loc, scale, clip=input_clip)
        dim = int(loc.numel())
        layers: list[nn.Module] = []
        prev = dim
        for width in hidden_layers:
            layers.append(nn.Linear(prev, int(width)))
            layers.append(activation_from_name(activation))
            if input_dropout > 0.0:
                layers.append(nn.Dropout(p=float(input_dropout)))
            prev = int(width)
        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        return self.forward_normalized(self.normalizer(x_raw))

    @torch.jit.export
    def forward_normalized(self, z: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.network(z))

    @torch.jit.export
    def forward_id(self, x_raw: torch.Tensor) -> torch.Tensor:
        return self.network(self.normalizer(x_raw))

    @torch.jit.export
    def forward_id_normalized(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)

    @torch.jit.export
    def normalize_input(self, x_raw: torch.Tensor) -> torch.Tensor:
        return self.normalizer.normalize(x_raw)

    @torch.jit.export
    def denormalize_input(self, z: torch.Tensor) -> torch.Tensor:
        return self.normalizer.denormalize(z)

    @torch.jit.export
    def input_scale(self) -> torch.Tensor:
        return self.normalizer.scale

    @torch.jit.export
    def input_loc(self) -> torch.Tensor:
        return self.normalizer.loc


def build_model(config: dict, loc: np.ndarray, scale: np.ndarray) -> CommittorNet:
    loc_t = torch.as_tensor(loc, dtype=torch.float32)
    scale_t = torch.as_tensor(scale, dtype=torch.float32)
    return CommittorNet(
        loc=loc_t,
        scale=scale_t,
        hidden_layers=list(config.get("hidden_layers", [64, 64])),
        activation=config.get("activation", "elu"),
        input_clip=float(config.get("input_clip", 0.0)),
        input_dropout=float(config.get("input_dropout", 0.0)),
    )


def save_model_metadata(path: str | Path, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def load_model(path: str | Path, device: torch.device | str):
    model = torch.jit.load(str(path), map_location=device)
    model.eval()
    return model
