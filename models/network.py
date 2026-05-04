import torch
import torch.nn as nn
import torch.nn.functional as F
from .detail_recovery import DetailRecoveryNet
from .grid_slicer import BilateralGridSlicer
from .sc_network import SpatialColorBilateralNet
from .temporal_network import TemporalBilateralNet

class BilateralGridDehazeNet(nn.Module):

    def __init__(self, backbone_channels: list[int] | None=None, color_bins: int=8, coeff_dim: int=12, num_frames: int=5, encode_size: int=256, detail_unshuffle: int=4, detail_channels: int=64, drop_path_rate: float=0.1, grid_dropout: float=0.1, temporal_attn_heads: int=4) -> None:
        super().__init__()
        if backbone_channels is None:
            backbone_channels = [24, 48, 96, 192]
        self.encode_size = encode_size
        self.num_frames = num_frames
        self.detail_unshuffle = detail_unshuffle
        self.sc_net = SpatialColorBilateralNet(backbone_channels=backbone_channels, color_bins=color_bins, coeff_dim=coeff_dim, drop_path_rate=drop_path_rate, grid_dropout=grid_dropout)
        self.t_net = TemporalBilateralNet(backbone_channels=backbone_channels, num_frames=num_frames, coeff_dim=coeff_dim, drop_path_rate=drop_path_rate, grid_dropout=grid_dropout, temporal_attn_heads=temporal_attn_heads)
        self.slicer = BilateralGridSlicer()
        self.detail_recovery = DetailRecoveryNet(unshuffle_factor=detail_unshuffle, base_channels=detail_channels)

    def _core_pipeline(self, hazy_frames: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        (B, T, C, H, W) = hazy_frames.shape
        center_idx = T // 2
        center_hazy = hazy_frames[:, center_idx]
        S = self.encode_size
        p = self.detail_unshuffle
        all_frames = hazy_frames.reshape(B * T, C, H, W)
        all_frames_lr = F.interpolate(all_frames, size=(S, S), mode='bilinear', align_corners=False).reshape(B, T, C, S, S)
        center_hazy_lr = all_frames_lr[:, center_idx]
        (sc_grid_r, sc_grid_g) = self.sc_net(center_hazy_lr)
        t_grid = self.t_net(all_frames_lr)
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        (mid_h, mid_w) = ((H + pad_h) // p, (W + pad_w) // p)
        center_hazy_mid = F.interpolate(center_hazy, size=(mid_h, mid_w), mode='bilinear', align_corners=False)
        lr_output = self.slicer.slice_and_apply(sc_grid_r, sc_grid_g, t_grid, center_hazy_mid, center_idx, T)
        output = self.detail_recovery(lr_output, center_hazy, H, W)
        grids = {'sc_grid_r': sc_grid_r, 'sc_grid_g': sc_grid_g, 't_grid': t_grid}
        return (output, grids)

    def forward(self, hazy_frames: torch.Tensor) -> dict[str, torch.Tensor]:
        (output, grids) = self._core_pipeline(hazy_frames)
        return {'output': output, 'grids': grids}

    @torch.no_grad()
    def inference(self, hazy_frames: torch.Tensor, target_h: int, target_w: int, encode_size: int | None=None) -> torch.Tensor:
        with torch.amp.autocast('cuda', dtype=torch.float16):
            (output, _) = self._core_pipeline(hazy_frames)
        if output.shape[-2] != target_h or output.shape[-1] != target_w:
            output = F.interpolate(output, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return output.clamp(0.0, 1.0)
