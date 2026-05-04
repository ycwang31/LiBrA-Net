import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import UNetBackbone

class SpatialColorBilateralNet(nn.Module):

    def __init__(self, backbone_channels: list[int] | None=None, color_bins: int=8, coeff_dim: int=12, drop_path_rate: float=0.1, grid_dropout: float=0.1) -> None:
        super().__init__()
        if backbone_channels is None:
            backbone_channels = [24, 48, 96, 192]
        self.color_bins = color_bins
        self.coeff_dim = coeff_dim
        self.backbone = UNetBackbone(channels=backbone_channels, drop_path_rate=drop_path_rate)
        feat_ch = backbone_channels[-1]
        out_ch = coeff_dim * color_bins * 2
        self.grid_head = nn.Sequential(nn.Conv2d(feat_ch, feat_ch, 3, padding=1, bias=True), nn.GELU(), nn.Dropout2d(p=grid_dropout), nn.Conv2d(feat_ch, out_ch, 1, bias=True))
        self._zero_init()

    def _zero_init(self) -> None:
        nn.init.zeros_(self.grid_head[-1].weight)
        nn.init.zeros_(self.grid_head[-1].bias)

    def forward(self, center_hazy_lr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = center_hazy_lr.shape[0]
        D = self.coeff_dim
        Gc = self.color_bins
        feat = self.backbone(center_hazy_lr)
        grid_out = self.grid_head(feat)
        (Gx, Gy) = (grid_out.shape[-2], grid_out.shape[-1])
        grid_out = grid_out.reshape(B, 2, D, Gc, Gx, Gy)
        sc_grid_r = grid_out[:, 0]
        sc_grid_g = grid_out[:, 1]
        return (sc_grid_r, sc_grid_g)
