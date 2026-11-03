"""Causal GRU representation with a scheduled gradient reversal domain head."""
from __future__ import annotations

import math

import torch
from torch import nn


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


def gradient_reverse(x: torch.Tensor, strength: float) -> torch.Tensor:
    return _GradientReversal.apply(x, strength)


def reversal_schedule(progress: float, maximum: float = 1.0) -> float:
    return maximum * (2 / (1 + math.exp(-10 * min(max(progress, 0), 1))) - 1)


# frame: every 10 Hz frame enters the GRU (pilot, 5/10/20 s).
# second: per-second summaries enter the GRU (09/10 v1/v2 runs; kept so they still load).
# window: window statistics of the per-second summaries plus a small GRU over 10 s blocks.
POOLINGS = ("frame", "second", "window")
CONTEXTS = {"frame": (5, 10, 20), "second": (20, 30, 60, 120), "window": (20, 30, 60, 120)}
FRAMES_PER_SECOND = 10


# Window-relative deviations are floored at this many train SDs so a flat signal stays near zero.
RELATIVE_SD_FLOOR = 0.1
SECOND_FEATURES = 46 + 31


def _windowed(values: torch.Tensor, masks: torch.Tensor, steps: int):
    batch, _, fields = values.shape
    shape = (batch, steps, FRAMES_PER_SECOND, fields)
    return values.reshape(shape), masks.reshape(shape)


def relative_summary(values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Per-second deviations from the driver's own statistics inside the causal window.

    Absolute eye openness or head angle differs a lot between people; drowsiness shows as
    long or frequent closures relative to that person's usual level (PERCLOS-like cues).
    Only frames inside the window ending at the current frame are used.
    """
    count = masks.sum(1, keepdim=True)
    center = (values * masks).sum(1, keepdim=True) / count.clamp_min(1)
    spread = (((values - center) ** 2 * masks).sum(1, keepdim=True) / count.clamp_min(1)).sqrt()
    z = ((values - center) / spread.clamp_min(RELATIVE_SD_FLOOR)).clamp(-6, 6) * masks
    steps = values.shape[1] // FRAMES_PER_SECOND
    zs, ms = _windowed(z, masks, steps)
    seconds = ms.sum(2)
    present = seconds > 0
    mean = zs.sum(2) / seconds.clamp_min(1)
    low = torch.where(present, torch.where(ms > 0, zs, torch.full_like(zs, 6.0)).amin(2), 0)
    high = torch.where(present, torch.where(ms > 0, zs, torch.full_like(zs, -6.0)).amax(2), 0)
    eyes = ms[..., 0] * ms[..., 1]
    eye_z = (zs[..., 0] + zs[..., 1]) / 2
    blink_z = (zs[..., 3] + zs[..., 4]) / 2
    eye_count = eyes.sum(2).clamp_min(1)
    closures = [((eye_z < -limit) * eyes).sum(2) / eye_count for limit in (1.0, 2.0)]
    closures += [((blink_z > limit) * eyes).sum(2) / eye_count for limit in (1.0, 2.0)]
    return torch.cat((mean, low, high, torch.stack(closures, -1)), -1)


def second_summary(values: torch.Tensor, masks: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
    """Per-second masked mean/std/min/max of normalized fields plus field and face coverage.

    Long contexts (30-120 s) carry slow drowsiness cues such as eye-closure duration and
    blink rate; summarizing each causal second keeps the GRU sequence short.
    """
    batch, frames, fields = values.shape
    steps = frames // FRAMES_PER_SECOND
    shape = (batch, steps, FRAMES_PER_SECOND, fields)
    v, m = values.reshape(shape), masks.reshape(shape)
    count = m.sum(2)
    present = count > 0
    mean = (v * m).sum(2) / count.clamp_min(1)
    std = (((v - mean[:, :, None]) ** 2 * m).sum(2) / count.clamp_min(1)).clamp_min(0).sqrt()
    large = torch.finfo(v.dtype).max
    low = torch.where(m > 0, v, torch.full_like(v, large)).amin(2)
    high = torch.where(m > 0, v, torch.full_like(v, -large)).amax(2)
    low, high = torch.where(present, low, 0), torch.where(present, high, 0)
    coverage = count / FRAMES_PER_SECOND
    face_coverage = face.reshape(batch, steps, FRAMES_PER_SECOND).mean(2, keepdim=True)
    return torch.cat((mean, std, low, high, coverage, face_coverage), -1)


BLOCK_SECONDS = 10


class CameraDANN(nn.Module):
    def __init__(self, n_classes: int, n_domains: int, hidden_size: int = 96,
                 layers: int = 2, dropout: float = 0.3, pooling: str = "frame"):
        super().__init__()
        if n_classes not in (2, 3) or n_domains < 1 or hidden_size < 1 or layers < 1:
            raise ValueError("Invalid DANN model dimensions")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0,1)")
        if pooling not in POOLINGS:
            raise ValueError("Pooling must be frame, second or window")
        self.pooling = pooling
        self.register_buffer("center", torch.zeros(9))
        self.register_buffer("scale", torch.ones(9))
        self.gru = nn.GRU(37 if pooling == "frame" else SECOND_FEATURES, hidden_size, num_layers=layers,
                          batch_first=True, dropout=dropout if layers > 1 else 0)
        if pooling == "window":
            # With ~40 drivers a learned bottleneck memorizes recordings; window statistics are
            # standardized per feature with train running statistics (like a StandardScaler) and
            # kept beside the block GRU states without a projection.
            self.block_norm = nn.BatchNorm1d(SECOND_FEATURES)
            self.block_dropout = nn.Dropout(dropout)
            self.pooled_norm = nn.BatchNorm1d(2 * SECOND_FEATURES)
            width = 2 * SECOND_FEATURES + 2 * hidden_size
            self.projection = nn.Identity()
        else:
            pooled = 2 * SECOND_FEATURES if pooling == "second" else 0
            self.pooled_norm = nn.Sequential(nn.LayerNorm(pooled), nn.Dropout(dropout)) if pooled else None
            # Index 3 stays a no-op so pilot checkpoints keep their parameter names.
            self.projection = nn.Sequential(nn.Linear(hidden_size * 2 + pooled, hidden_size),
                                            nn.LayerNorm(hidden_size), nn.ReLU(), nn.Identity())
            width = hidden_size
        self.width = width
        # Dropout sits on the class path only: the adversary must see the full shared features.
        self.class_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(width, n_classes)
        domain_width = max(hidden_size, 64) if pooling == "window" else hidden_size
        self.domain_head = nn.Sequential(nn.Linear(width, domain_width), nn.ReLU(),
                                         nn.Dropout(dropout), nn.Linear(domain_width, n_domains))

    def representation(self, x: torch.Tensor) -> torch.Tensor:
        masks = x[:, :, 9:18] * x[:, :, -1:]
        values = ((x[:, :, :9] - self.center) / self.scale).clamp(-12, 12) * masks
        if self.pooling == "frame":
            adjacent = torch.zeros_like(masks)
            adjacent[:, 1:] = masks[:, 1:] * masks[:, :-1]
            differences = torch.zeros_like(values)
            differences[:, 1:] = (values[:, 1:] - values[:, :-1]) * adjacent[:, 1:]
            sequence = torch.cat((values, x[:, :, 9:], differences, adjacent), -1)
            states, _ = self.gru(sequence)
            return self.projection(torch.cat((states[:, -1], states.mean(1)), -1))
        if x.shape[1] % (FRAMES_PER_SECOND * (BLOCK_SECONDS if self.pooling == "window" else 1)):
            raise ValueError("Summary pooling needs whole seconds (window: whole 10 s blocks) of history")
        summary = torch.cat((second_summary(values, masks, x[:, :, -1]), relative_summary(values, masks)), -1)
        statistics = torch.cat((summary.mean(1), summary.std(1)), -1)
        if self.pooling == "second":
            states, _ = self.gru(summary)
            return self.projection(torch.cat((states[:, -1], states.mean(1), self.pooled_norm(statistics)), -1))
        blocks = self.block_norm(summary.transpose(1, 2)).transpose(1, 2)
        blocks = blocks.reshape(blocks.shape[0], -1, BLOCK_SECONDS, blocks.shape[2]).mean(2)
        states, _ = self.gru(self.block_dropout(blocks))
        return torch.cat((self.pooled_norm(statistics), states[:, -1], states.mean(1)), -1)

    def forward(self, x: torch.Tensor, reversal: float = 0.0):
        shared = self.representation(x)
        return self.classifier(self.class_dropout(shared)), self.domain_head(gradient_reverse(shared, reversal))


class ClassifierExport(nn.Module):
    """Inference graph includes learned training normalization and class probabilities."""
    def __init__(self, model: CameraDANN):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model.classifier(self.model.representation(x)), dim=-1)
