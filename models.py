from __future__ import annotations

from typing import Literal
import torch
import torch.nn as nn


class MLPBackbone(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int = 128, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = input_dim * seq_len
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.chomp_size == 0 else x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + self.downsample(x))


class TCNBackbone(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int = 128, depth: int = 4, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i in range(depth):
            layers.append(TemporalBlock(in_ch, hidden_dim, kernel_size, 2 ** i, dropout))
            in_ch = hidden_dim
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x.transpose(1, 2))
        return z[:, :, -1]


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        hidden_dim: int = 32,
        depth: int = 1,
        nhead: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_dim = hidden_dim

    def forward(self, x):
        T = x.size(1)
        z = self.input_proj(x) + self.pos[:, :T, :]
        z = self.encoder(z)
        z = self.norm(z)
        return z.mean(dim=1)


def build_backbone(backbone: Literal["mlp", "tcn", "transformer"], input_dim: int, seq_len: int, hidden_dim: int, depth: int, dropout: float):
    if backbone == "mlp":
        return MLPBackbone(input_dim, seq_len, hidden_dim, depth, dropout)
    if backbone == "tcn":
        return TCNBackbone(input_dim, seq_len, hidden_dim, depth, 3, dropout)
    if backbone == "transformer":
        return TransformerBackbone(input_dim, seq_len, hidden_dim, depth, 4, dropout)
    raise ValueError(f"Unknown backbone: {backbone}")


class DirectForecastModel(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.out_dim, 1)

    def forward(self, x: torch.Tensor, p_clear: torch.Tensor | None = None, x_phys: torch.Tensor | None = None) -> dict:
        h = self.backbone(x)
        return {"y_hat": self.head(h).squeeze(-1)}


class DecompositionForecastModel(nn.Module):
    def __init__(self, backbone: nn.Module, c_max: float = 1.2, residual_scale: float = 1.0):
        super().__init__()
        self.backbone = backbone
        self.c_head = nn.Linear(backbone.out_dim, 1)
        self.r_head = nn.Linear(backbone.out_dim, 1)
        self.c_max = float(c_max)
        self.residual_scale = float(residual_scale)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, p_clear: torch.Tensor, x_phys: torch.Tensor | None = None) -> dict:
        h = self.backbone(x)
        c_hat = self.c_max * torch.sigmoid(self.c_head(h).squeeze(-1))
        r_hat = self.residual_scale * self.r_head(h).squeeze(-1)
        y_hat = self.alpha * p_clear * c_hat + r_hat
        return {"y_hat": y_hat, "c_hat": c_hat, "r_hat": r_hat}


class ClearSkyFiLMForecastModel(nn.Module):
    """
    Weather encoder + clear-sky physics FiLM.

    x      : weather sequence, shape [B, T, D_w]
    x_phys : physics/metadata sequence, shape [B, T, D_p]
             Only the last step is used for conditioning because the target is the
             last step of the input window.
    """
    def __init__(
        self,
        backbone: nn.Module,
        physics_input_dim: int,
        film_hidden_dim: int | None = None,
        modulation_scale: float = 0.1,
    ):
        super().__init__()
        if physics_input_dim <= 0:
            raise ValueError("clear_sky_film requires physics_input_dim > 0")

        self.backbone = backbone
        self.physics_input_dim = int(physics_input_dim)
        self.modulation_scale = float(modulation_scale)
        film_hidden_dim = int(film_hidden_dim or backbone.out_dim)

        self.phys_encoder = nn.Sequential(
            nn.Linear(self.physics_input_dim, film_hidden_dim),
            nn.ReLU(),
            nn.Linear(film_hidden_dim, 2 * backbone.out_dim),
        )
        self.norm = nn.LayerNorm(backbone.out_dim)
        self.head = nn.Linear(backbone.out_dim, 1)

        # Start close to the direct backbone. This reduces the chance that FiLM
        # destroys a good weather representation early in training.
        nn.init.zeros_(self.phys_encoder[-1].weight)
        nn.init.zeros_(self.phys_encoder[-1].bias)

    def forward(self, x: torch.Tensor, p_clear: torch.Tensor | None = None, x_phys: torch.Tensor | None = None) -> dict:
        if x_phys is None:
            raise ValueError("clear_sky_film requires x_phys in the batch")

        h = self.backbone(x)

        z = x_phys[:, -1, :] if x_phys.dim() == 3 else x_phys
        gamma_beta = self.phys_encoder(z)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        gamma = self.modulation_scale * torch.tanh(gamma)
        beta = self.modulation_scale * torch.tanh(beta)

        h_mod = self.norm((1.0 + gamma) * h + beta)
        y_hat = self.head(h_mod).squeeze(-1)

        return {
            "y_hat": y_hat,
            "film_gamma_mean": gamma.mean(dim=-1),
            "film_beta_mean": beta.mean(dim=-1),
        }


def build_model(
    setting: str,
    backbone_name: str,
    input_dim: int,
    seq_len: int,
    hidden_dim: int = 128,
    depth: int = 3,
    dropout: float = 0.1,
    c_max: float = 1.2,
    residual_scale: float = 1.0,
    physics_input_dim: int = 0,
    film_modulation_scale: float = 0.1,
):
    backbone = build_backbone(backbone_name, input_dim, seq_len, hidden_dim, depth, dropout)
    if setting in ["direct", "physics_feature"]:
        return DirectForecastModel(backbone)
    if setting == "decomposition":
        return DecompositionForecastModel(backbone, c_max, residual_scale)
    if setting == "clear_sky_film":
        return ClearSkyFiLMForecastModel(
            backbone=backbone,
            physics_input_dim=physics_input_dim,
            film_hidden_dim=hidden_dim,
            modulation_scale=film_modulation_scale,
        )
    raise ValueError(f"Unknown setting: {setting}")
