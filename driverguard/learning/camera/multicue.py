"""Camera-only temporal fusion of eyes, mouth and head motion, with ablations.

Input stays on the existing 10 Hz geometry/mask contract. Differences are
per sample (not blink velocity); missing measurements never create motion.
"""
from __future__ import annotations

import torch
from torch import nn

MULTICUE_VERSION = "masked_branch_gru_differences_v1"
CUE_FIELDS = {"eyes": (0, 1, 3, 4), "mouth": (2, 5), "head": (6, 7, 8)}
CUE_SETS = {
    "full": ("eyes", "mouth", "head"),
    "eyes_mouth": ("eyes", "mouth"),
    "head": ("head",),
    "quality_only": ("quality",),
}


def cue_sequence(x: torch.Tensor, cue: str) -> torch.Tensor:
    """Past-only values, masks, differences and masks for valid adjacent pairs."""
    if cue == "quality":
        return x[:, :, 9:]
    indices = CUE_FIELDS[cue]
    masks = x[:, :, [i + 9 for i in indices]] * x[:, :, -1:]
    values = x[:, :, list(indices)] * masks
    adjacent = torch.zeros_like(masks)
    adjacent[:, 1:] = masks[:, 1:] * masks[:, :-1]
    differences = torch.zeros_like(values)
    differences[:, 1:] = (values[:, 1:] - values[:, :-1]) * adjacent[:, 1:]
    return torch.cat((values, masks, differences, adjacent), dim=-1)


class CameraMultiCueGRU(nn.Module):
    """Separate temporal encoders, followed by learned joint evidence fusion."""

    def __init__(self, cue_set: str = "full", hidden_size: int = 16, dropout: float = 0.3):
        super().__init__()
        if cue_set not in CUE_SETS:
            raise ValueError("Unknown camera cue set")
        self.cue_set = cue_set
        self.branches = nn.ModuleDict({
            cue: nn.GRU(10 if cue == "quality" else 4 * len(CUE_FIELDS[cue]),
                        hidden_size, batch_first=True)
            for cue in CUE_SETS[cue_set]
        })
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size * len(self.branches), hidden_size),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states = [branch(cue_sequence(x, cue))[0][:, -1] for cue, branch in self.branches.items()]
        return self.head(torch.cat(states, dim=-1)).squeeze(-1)
