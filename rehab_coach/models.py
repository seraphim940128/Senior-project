from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class DSConv(nn.Module):
    def __init__(self, channels: int = 960, mid: int = 256, p: float = 0.1):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, groups=channels)
        self.pw = nn.Conv2d(channels, mid, kernel_size=1)
        self.bn = nn.GroupNorm(16, mid)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x.squeeze(-1).squeeze(-1)


class TinyTFClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        mid: int = 256,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        use_delta: bool = True,
        max_len: int = 300,
    ):
        super().__init__()
        self.use_delta = use_delta
        self.roi_head = DSConv(channels=960, mid=mid, p=dropout)
        in_dim = mid * 2 if use_delta else mid
        self.proj = nn.Linear(in_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.tf = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.att = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            batch, seq_len, channels, height, width = x.shape
            x = x.view(batch * seq_len, channels, height, width)
            x = self.roi_head(x)
            x = x.view(batch, seq_len, -1)
        elif x.ndim != 3:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        x = x - x.mean(dim=1, keepdim=True)
        x = x / x.std(dim=1, keepdim=True).clamp_min(1e-6)

        if self.use_delta:
            dx = torch.diff(x, dim=1, prepend=x[:, :1, :])
            x = torch.cat([x, dx], dim=2)

        x = self.proj(x)
        seq_len = x.size(1)
        x = x + self.pos[:, :seq_len, :]

        encoded = self.tf(x)
        weights = torch.softmax(self.att(encoded).squeeze(-1), dim=1)
        pooled = (encoded * weights.unsqueeze(-1)).sum(dim=1)
        return self.fc(pooled)


class OnlineFeatureBuffer:
    def __init__(self, sequence_len: int = 30):
        self.sequence_len = sequence_len
        self.buffer = []

    def push(self, feature: torch.Tensor) -> None:
        self.buffer.append(feature.detach().cpu())
        if len(self.buffer) > self.sequence_len:
            self.buffer.pop(0)

    def ready(self) -> bool:
        return len(self.buffer) > 0

    def last(self) -> torch.Tensor:
        return self.buffer[-1]

    def tensor(self, device: torch.device) -> torch.Tensor:
        if not self.buffer:
            return torch.zeros(
                (1, self.sequence_len, 960, 3, 3),
                dtype=torch.float32,
                device=device,
            )
        sequence = self.buffer[-self.sequence_len :]
        if len(sequence) < self.sequence_len:
            sequence = [sequence[0]] * (self.sequence_len - len(sequence)) + sequence
        return torch.stack(sequence, dim=0).unsqueeze(0).to(device)


def _clean_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state:
        return state
    if all(key.startswith("module.") for key in state):
        return {key.replace("module.", "", 1): value for key, value in state.items()}
    return state


def load_classifier(weights_path: str, num_classes: int, device: torch.device) -> TinyTFClassifier:
    model = TinyTFClassifier(num_classes=num_classes)
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    checkpoint = _clean_state_dict(checkpoint)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model
