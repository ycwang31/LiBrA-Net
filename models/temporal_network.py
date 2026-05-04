import torch
import torch.nn as nn
from .backbone import UNetBackbone

class TemporalAttention(nn.Module):

    def __init__(self, channels: int, num_heads: int=4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, channels * 2), nn.GELU(), nn.Linear(channels * 2, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, T, C, H, W) = x.shape
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        normed = self.norm1(x_flat)
        (attn_out, _) = self.attn(normed, normed, normed)
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.ffn(self.norm2(x_flat))
        return x_flat.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2)

class TemporalBilateralNet(nn.Module):

    def __init__(self, backbone_channels: list[int] | None=None, num_frames: int=5, coeff_dim: int=12, drop_path_rate: float=0.1, grid_dropout: float=0.1, temporal_attn_heads: int=4) -> None:
        super().__init__()
        if backbone_channels is None:
            backbone_channels = [24, 48, 96, 192]
        self.num_frames = num_frames
        self.coeff_dim = coeff_dim
        self.backbone = UNetBackbone(channels=backbone_channels, drop_path_rate=drop_path_rate)
        feat_ch = backbone_channels[-1]
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, feat_ch, 1, 1))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        self.temporal_attn = TemporalAttention(channels=feat_ch, num_heads=temporal_attn_heads)
        self.grid_head = nn.Sequential(nn.Conv2d(feat_ch, feat_ch, 3, padding=1, bias=True), nn.GELU(), nn.Dropout2d(p=grid_dropout), nn.Conv2d(feat_ch, coeff_dim, 1, bias=True))
        self._zero_init()

    def _zero_init(self) -> None:
        nn.init.zeros_(self.grid_head[-1].weight)
        nn.init.zeros_(self.grid_head[-1].bias)

    def forward(self, all_frames_lr: torch.Tensor) -> torch.Tensor:
        (B, T, C, S_h, S_w) = all_frames_lr.shape
        D = self.coeff_dim
        flat = all_frames_lr.reshape(B * T, C, S_h, S_w)
        frame_feats = self.backbone(flat)
        (feat_ch, Gx, Gy) = (frame_feats.shape[1], frame_feats.shape[2], frame_feats.shape[3])
        frame_feats = frame_feats.reshape(B, T, feat_ch, Gx, Gy)
        frame_feats = frame_feats + self.temporal_pos_embed
        frame_feats = self.temporal_attn(frame_feats)
        per_frame = frame_feats.reshape(B * T, feat_ch, Gx, Gy)
        per_frame_grids = self.grid_head(per_frame)
        t_grid = per_frame_grids.reshape(B, T, D, Gx, Gy).permute(0, 2, 1, 3, 4)
        return t_grid
