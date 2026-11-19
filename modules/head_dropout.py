"""NoImageHeadDropout: NoImageHead with Dropout2d in the backbone for proper MC-Dropout training."""

import torch
import torch.nn as nn


class NoImageHeadDropout(nn.Module):
    """
    Identical to NoImageHead but with nn.Dropout2d inserted after each ReLU in
    the backbone.  Training with dropout active means the MC-Dropout inference
    mode (head.train() + K forward passes) forms a proper variational posterior
    rather than a post-hoc approximation.

    Architecture:
        Conv(3 -> H, 3x3, pad=1) + ReLU + Dropout2d(p)
        Conv(H -> H, 3x3, pad=1) + ReLU + Dropout2d(p)
        Conv(H -> 1, 1x1)   <- Laplace target (final_conv)

    The final_conv has no dropout — it is the only Bayesianized layer.
    Dropout is active when the module is in train() mode and off in eval() mode.
    """

    def __init__(
        self,
        channels_in: int = 3,
        hidden_channels: int = 16,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        H = hidden_channels
        p = dropout_rate

        self.backbone = nn.Sequential(
            nn.Conv2d(channels_in, H, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p),
            nn.Conv2d(H, H, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p),
        )

        # Final prediction layer — only this layer is Bayesianized by Laplace.
        # Dropout is intentionally absent here so Laplace can be fitted cleanly.
        self.final_conv = nn.Conv2d(H, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) — [sam_logits, pos_map, neg_map]
        Returns:
            (B, 1, H, W) refined logits
        """
        return self.final_conv(self.backbone(x))

    def final_layer_parameters(self):
        """Return parameters of final_conv only (for Laplace)."""
        return list(self.final_conv.parameters())
