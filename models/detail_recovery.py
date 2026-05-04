import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.gelu(self.conv1(x))
        out = self.conv2(out)
        return F.gelu(out + residual)

class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int=3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))

class SEBlock(nn.Module):

    def __init__(self, channels: int, reduction: int=4) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, mid, 1, bias=True)
        self.fc2 = nn.Conv2d(mid, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.gelu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class DetailRecoveryNet(nn.Module):

    def __init__(self, unshuffle_factor: int=4, base_channels: int=64) -> None:
        super().__init__()
        self.unshuffle_factor = unshuffle_factor
        in_ch = 3 * unshuffle_factor ** 2
        out_ch = in_ch
        C = base_channels
        self.detail_enc = nn.Sequential(nn.Conv2d(in_ch, C, 3, 1, 1), nn.GELU(), ResBlock(C), ResBlock(C), DepthwiseSeparableConv(C, C), nn.GELU())
        self.struct_enc = nn.Sequential(nn.Conv2d(3, C, 3, 1, 1), nn.GELU(), ResBlock(C))
        self.fuse = nn.Sequential(nn.Conv2d(C * 2, C, 1), nn.GroupNorm(num_groups=8, num_channels=C), nn.GELU(), SEBlock(C), ResBlock(C), ResBlock(C), nn.Conv2d(C, out_ch, 3, 1, 1))
        self.pixel_shuffle = nn.PixelShuffle(unshuffle_factor)
        self._init_zero_residual()

    def _init_zero_residual(self) -> None:
        last_conv = self.fuse[-1]
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

    def forward(self, lr_dehazed: torch.Tensor, hr_hazy: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        p = self.unshuffle_factor
        pad_h = (p - target_h % p) % p
        pad_w = (p - target_w % p) % p
        if pad_h > 0 or pad_w > 0:
            hr_hazy = F.pad(hr_hazy, (0, pad_w, 0, pad_h), mode='reflect')
        detail_input = F.pixel_unshuffle(hr_hazy, p)
        detail_feat = self.detail_enc(detail_input)
        (mid_h, mid_w) = detail_feat.shape[-2:]
        struct_feat = self.struct_enc(lr_dehazed)
        if struct_feat.shape[-2:] != (mid_h, mid_w):
            struct_feat = F.interpolate(struct_feat, size=(mid_h, mid_w), mode='bilinear', align_corners=False)
        fused = self.fuse(torch.cat([detail_feat, struct_feat], dim=1))
        residual = self.pixel_shuffle(fused)
        padded_size = hr_hazy.shape[-2:]
        hazy_down = F.interpolate(hr_hazy, size=lr_dehazed.shape[-2:], mode='bilinear', align_corners=False)
        delta_lr = lr_dehazed - hazy_down
        dehaze_delta = F.interpolate(delta_lr, size=padded_size, mode='bilinear', align_corners=False)
        output = (hr_hazy + dehaze_delta + residual).clamp(0.0, 1.0)
        return output[:, :, :target_h, :target_w]
