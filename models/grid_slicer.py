import torch
import torch.nn as nn
import torch.nn.functional as F
from .lie_group import apply_affine, cayley_affine

class BilateralGridSlicer(nn.Module):

    def slice_and_apply(self, sc_grid_r: torch.Tensor, sc_grid_g: torch.Tensor, t_grid: torch.Tensor, hazy_image: torch.Tensor, center_t_idx: int, num_frames: int) -> torch.Tensor:
        (B, _, H, W) = hazy_image.shape
        sc_coeffs_r = self._slice_sc_grid(sc_grid_r, hazy_image[:, 0], H, W)
        sc_coeffs_g = self._slice_sc_grid(sc_grid_g, hazy_image[:, 1], H, W)
        sc_coeffs = (sc_coeffs_r + sc_coeffs_g) / 2
        t_coeffs = self._slice_t_grid(t_grid, center_t_idx, num_frames, H, W)
        raw_coeffs = sc_coeffs + t_coeffs
        raw_flat = raw_coeffs.permute(0, 2, 3, 1)
        affine_flat = cayley_affine(raw_flat)
        affine_coeffs = affine_flat.permute(0, 3, 1, 2)
        return apply_affine(affine_coeffs, hazy_image)

    def _slice_sc_grid(self, sc_grid: torch.Tensor, guide_channel: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B = sc_grid.shape[0]
        y_coords = torch.linspace(-1, 1, H, device=sc_grid.device, dtype=sc_grid.dtype)
        x_coords = torch.linspace(-1, 1, W, device=sc_grid.device, dtype=sc_grid.dtype)
        (grid_y, grid_x) = torch.meshgrid(y_coords, x_coords, indexing='ij')
        color_norm = guide_channel * 2 - 1
        spatial_x = grid_x.unsqueeze(0).expand(B, -1, -1)
        spatial_y = grid_y.unsqueeze(0).expand(B, -1, -1)
        grid = torch.stack([spatial_x, spatial_y, color_norm], dim=-1)
        grid = grid.unsqueeze(1)
        out = F.grid_sample(sc_grid, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return out.squeeze(2)

    def _slice_t_grid(self, t_grid: torch.Tensor, center_t_idx: int, num_frames: int, H: int, W: int) -> torch.Tensor:
        B = t_grid.shape[0]
        T = num_frames
        y_coords = torch.linspace(-1, 1, H, device=t_grid.device, dtype=t_grid.dtype)
        x_coords = torch.linspace(-1, 1, W, device=t_grid.device, dtype=t_grid.dtype)
        (grid_y, grid_x) = torch.meshgrid(y_coords, x_coords, indexing='ij')
        spatial_x = grid_x.unsqueeze(0).expand(B, -1, -1)
        spatial_y = grid_y.unsqueeze(0).expand(B, -1, -1)
        if T <= 1:
            t_norm = 0.0
        else:
            t_norm = 2.0 * center_t_idx / (T - 1) - 1.0
        t_val = torch.full_like(spatial_x, t_norm)
        grid = torch.stack([spatial_x, spatial_y, t_val], dim=-1)
        grid = grid.unsqueeze(1)
        out = F.grid_sample(t_grid, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return out.squeeze(2)
