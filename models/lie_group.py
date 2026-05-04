import torch
import torch.nn as nn

def cayley_affine(raw_coeffs: torch.Tensor) -> torch.Tensor:
    shape = raw_coeffs.shape[:-1]
    M = raw_coeffs[..., :9].reshape(*shape, 3, 3)
    b = raw_coeffs[..., 9:]
    I = torch.eye(3, device=M.device, dtype=M.dtype).expand_as(M)
    lhs = I + M / 2
    rhs = I - M / 2
    A = torch.linalg.solve(rhs, lhs)
    return torch.cat([A.reshape(*shape, 9), b], dim=-1)

def apply_affine(coeffs: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    A = coeffs[:, :9].reshape(coeffs.shape[0], 3, 3, coeffs.shape[2], coeffs.shape[3])
    bias = coeffs[:, 9:]
    out = torch.einsum('bcjhw,bjhw->bchw', A, image) + bias
    return out.clamp(0.0, 1.0)

class LieRegularization(nn.Module):

    def __init__(self, smooth_weight: float=0.05, identity_weight: float=0.01, s3_weight: float=0.02, temporal_weight: float=0.08) -> None:
        super().__init__()
        self.smooth_w = smooth_weight
        self.identity_w = identity_weight
        self.s3_w = s3_weight
        self.temporal_w = temporal_weight

    def forward(self, grids: dict[str, torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(iter(grids.values())).device)
        sc_r = grids['sc_grid_r']
        sc_g = grids['sc_grid_g']
        t_g = grids['t_grid']
        for grid in (sc_r, sc_g, t_g):
            dx = (grid[..., 1:, :] - grid[..., :-1, :]).pow(2).mean()
            dy = (grid[..., :, 1:] - grid[..., :, :-1]).pow(2).mean()
            loss = loss + self.smooth_w * (dx + dy)
        for grid in (sc_r, sc_g, t_g):
            M_part = grid[:, :9]
            loss = loss + self.identity_w * M_part.pow(2).mean()
        loss = loss + self.s3_w * (sc_r - sc_g).abs().mean()
        if t_g.shape[2] > 1:
            dt = (t_g[:, :, 1:] - t_g[:, :, :-1]).pow(2).mean()
            loss = loss + self.temporal_w * dt
        return loss
