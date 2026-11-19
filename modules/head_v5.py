"""
v5 head: a wider, dilated corrective head.

Motivation
----------
`NoImageHead` is 2,785 params with a 3x3 receptive field of 5 px, and it buys
only +0.012 IoU over raw SAM (0.6196 -> 0.6312 on test). Because the training
loop is I/O bound (17.7 ms GPU vs 922 ms dataloader per step), that capacity is
not buying anything in wall clock either: a 28.7k-param dilated head costs
62 ms/step on GPU, still far below even the *fixed* data budget.

Dilations 1/2/4/1 over four 3x3 convs give a 23 px receptive field, enough to
reason about boundary context rather than single-pixel logit corrections.

Attribute names `backbone` / `final_conv` are kept identical to `NoImageHead`
so every downstream consumer (Laplace fitting, active_prompting, checkpointing)
works unchanged.
"""

from typing import Sequence

import torch
import torch.nn as nn


class DilatedHeadV5(nn.Module):
    """
    Input  : (B, 3, H, W) — [sam_logits, pos_map, neg_map]
    Output : (B, 1, H, W) — refined logits

    backbone   : 4 x [Conv3x3(dilated) + ReLU]  (+ optional Dropout2d)
    final_conv : Conv 1x1, C -> 1               <- Laplace target (last-layer mode)
    """

    def __init__(
        self,
        channels_in: int = 3,
        hidden_channels: int = 32,
        dilations: Sequence[int] = (1, 2, 4, 1),
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        C = hidden_channels
        layers = []
        c_prev = channels_in
        for d in dilations:
            layers += [
                nn.Conv2d(c_prev, C, kernel_size=3, padding=d, dilation=d),
                nn.ReLU(inplace=True),
            ]
            if dropout_rate > 0:
                layers.append(nn.Dropout2d(p=dropout_rate))
            c_prev = C
        self.backbone = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(C, 1, kernel_size=1, bias=True)

        self.hidden_channels = C
        self.dilations = tuple(dilations)
        self.dropout_rate = dropout_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_conv(self.backbone(x))

    def final_layer_parameters(self):
        return list(self.final_conv.parameters())

    @property
    def receptive_field(self) -> int:
        """Odd-sized RF in pixels for the stacked dilated 3x3 convs."""
        rf = 1
        for d in self.dilations:
            rf += 2 * d
        return rf


def build_head(hcfg: dict) -> nn.Module:
    """
    Construct a head from the `head:` block of a config.

    variant: "no_image"  -> original NoImageHead (2,785 params)
             "dilated_v5" -> DilatedHeadV5       (28,673 params @ C=32)
    """
    variant = hcfg.get("variant", "dilated_v5")
    if variant == "no_image":
        from modules.head import NoImageHead
        return NoImageHead(
            channels_in=hcfg.get("channels_in", 3),
            hidden_channels=hcfg.get("hidden_channels", 16),
        )
    if variant == "dilated_v5":
        return DilatedHeadV5(
            channels_in=hcfg.get("channels_in", 3),
            hidden_channels=hcfg.get("hidden_channels", 32),
            dilations=hcfg.get("dilations", [1, 2, 4, 1]),
            dropout_rate=hcfg.get("dropout_rate", 0.0),
        )
    raise ValueError(f"Unknown head variant: {variant!r}. Use 'no_image' or 'dilated_v5'.")
