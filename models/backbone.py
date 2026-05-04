import torch
import torch.nn as nn
import torch.nn.functional as F

class DropPath(nn.Module):

    def __init__(self, drop_prob: float=0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep

class ResBlock(nn.Module):

    def __init__(self, channels: int, drop_path_rate: float=0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.drop_path(self.conv2(F.gelu(self.conv1(x)))) + x)

class DownBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, drop_path_rate: float=0.0) -> None:
        super().__init__()
        num_groups = min(32, out_ch // 4) if out_ch >= 16 else out_ch
        self.down = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False), nn.GroupNorm(num_groups, out_ch), nn.GELU(), ResBlock(out_ch, drop_path_rate=drop_path_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)

class UNetBackbone(nn.Module):

    def __init__(self, channels: list[int] | None=None, drop_path_rate: float=0.1) -> None:
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]
        assert len(channels) == 4, 'Expected 4 channel stages'
        (c0, c1, c2, c3) = channels
        dp = drop_path_rate
        rates = [dp * i / 5 for i in range(6)]
        self.stem = DownBlock(3, c0, drop_path_rate=rates[0])
        self.down1 = DownBlock(c0, c1, drop_path_rate=rates[1])
        self.down2 = DownBlock(c1, c2, drop_path_rate=rates[2])
        self.down3 = DownBlock(c2, c3, drop_path_rate=rates[3])
        self.bottleneck = nn.Sequential(ResBlock(c3, drop_path_rate=rates[4]), ResBlock(c3, drop_path_rate=rates[5]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.stem(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        feat = self.bottleneck(e3)
        return feat
