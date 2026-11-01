from __future__ import annotations

import torch
from torch import nn


class VisualPatchTransformer(nn.Module):
    """Patchify each paired scene/gaze frame, retain four spatial tokens per frame."""

    def __init__(self, timesteps: int = 30, dim: int = 128, heads: int = 4, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.patch = nn.Conv2d(2, dim, kernel_size=16, stride=16)
        self.spatial_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.time_position = nn.Parameter(torch.zeros(1, timesteps, 1, dim))
        self.space_position = nn.Parameter(torch.zeros(1, 1, 4, dim))
        encoder = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, scene_gaze: torch.Tensor) -> torch.Tensor:
        if scene_gaze.ndim != 5 or scene_gaze.shape[1:3] != (2, self.timesteps):
            raise ValueError(f"Expected scene/gaze [B,2,{self.timesteps},H,W]")
        batch, _, steps, height, width = scene_gaze.shape
        frames = scene_gaze.permute(0, 2, 1, 3, 4).reshape(batch * steps, 2, height, width)
        patches = self.spatial_pool(self.patch(frames)).flatten(2).transpose(1, 2)
        tokens = patches.reshape(batch, steps, 4, -1) + self.time_position + self.space_position
        encoded = self.encoder(tokens.reshape(batch, steps * 4, -1))
        return self.norm(encoded.mean(dim=1))


class SignalTimeTransformer(nn.Module):
    def __init__(self, timesteps: int = 30, signal_dim: int = 6, dim: int = 64, heads: int = 4, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.input_projection = nn.Linear(signal_dim, dim)
        self.position = nn.Parameter(torch.zeros(1, timesteps, dim))
        encoder = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        if signals.ndim != 3 or signals.shape[1] != self.timesteps:
            raise ValueError(f"Expected signals [B,{self.timesteps},6]")
        tokens = self.input_projection(signals) + self.position
        return self.norm(self.encoder(tokens).mean(dim=1))


class TransformerTakeoverModel(nn.Module):
    def __init__(self, timesteps: int = 30, signal_dim: int = 6, hmi_dim: int = 9) -> None:
        super().__init__()
        self.visual_branch = VisualPatchTransformer(timesteps=timesteps)
        self.signal_branch = SignalTimeTransformer(timesteps=timesteps, signal_dim=signal_dim)
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64 + hmi_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 1),
        )

    def forward(self, scene_gaze: torch.Tensor, signals: torch.Tensor, hmi: torch.Tensor) -> torch.Tensor:
        if hmi.ndim != 2 or hmi.shape[1] != 9:
            raise ValueError("Expected HMI [B,9]")
        return self.classifier(torch.cat((self.visual_branch(scene_gaze), self.signal_branch(signals), hmi), dim=1)).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, scene_gaze: torch.Tensor, signals: torch.Tensor, hmi: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(scene_gaze, signals, hmi))
