import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_DINOV2_REPO_DIR = Path(__file__).resolve().parent / 'pretrained' / 'facebookresearch_dinov2_main'
DEFAULT_DINOV2_WEIGHTS_PATH = Path(__file__).resolve().parent / 'pretrained' / 'dinov2_vits14_pretrain.pth'

class DINOv2FeatureExtractor(nn.Module):

    def __init__(self, weights_path: str | Path=DEFAULT_DINOV2_WEIGHTS_PATH, repo_dir: str | Path | None=None) -> None:
        super().__init__()
        resolved_repo_dir = Path(repo_dir or os.environ.get('DINOV2_REPO_DIR', DEFAULT_DINOV2_REPO_DIR))
        self.model = torch.hub.load(str(resolved_repo_dir), 'dinov2_vits14', pretrained=False, source='local')
        state_dict = torch.load(Path(weights_path), map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict)
        for param in self.model.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = (x - self.mean) / self.std
        (_, _, H, W) = x.shape
        patch_size = 14
        new_H = H // patch_size * patch_size
        new_W = W // patch_size * patch_size
        if new_H != H or new_W != W:
            x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
        features = self.model.get_intermediate_layers(x, n=[2, 5, 8, 11], reshape=True)
        return list(features)

def charbonnier_loss(pred: torch.Tensor, gt: torch.Tensor, eps: float=0.001) -> torch.Tensor:
    return torch.sqrt((pred - gt) ** 2 + eps ** 2).mean()

def perceptual_loss(feat_extractor: DINOv2FeatureExtractor, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pred_feats = feat_extractor(pred)
    gt_feats = feat_extractor(gt)
    loss = torch.tensor(0.0, device=pred.device)
    for (pf, gf) in zip(pred_feats, gt_feats):
        loss = loss + F.l1_loss(pf, gf)
    return loss / len(pred_feats)

def scale_separation_loss(large_output: torch.Tensor, small_output: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    (H, W) = gt.shape[-2:]
    gt_low = F.interpolate(F.interpolate(gt, size=(H // 8, W // 8), mode='bilinear', align_corners=False), size=(H, W), mode='bilinear', align_corners=False)
    loss_low = charbonnier_loss(large_output, gt_low)
    loss_high = charbonnier_loss(small_output, gt)
    return loss_low + loss_high

def regularization_loss(grids: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
    loss = torch.tensor(0.0, device=grids['small']['feat'].device)
    for scale in ['small', 'large']:
        feat = grids[scale]['feat']
        alpha = grids[scale]['alpha']
        loss = loss + feat.abs().mean()
        loss = loss + alpha.abs().mean() * 0.1
        if 'r' in grids[scale]:
            r = grids[scale]['r']
            loss = loss + ((r - 3.0) ** 2).mean() * 0.1
    return loss

class DehazeLoss(nn.Module):

    def __init__(self, pixel_weight: float=1.0, perc_weight: float=0.04, lie_weight: float=0.3, charbonnier_eps: float=0.001, lie_temporal_weight: float=0.08, lie_smooth_weight: float=0.05, lie_identity_weight: float=0.01, lie_s3_weight: float=0.02) -> None:
        super().__init__()
        self.pixel_weight = pixel_weight
        self.perc_weight = perc_weight
        self.lie_weight = lie_weight
        self.charbonnier_eps = charbonnier_eps
        self.feat_extractor = DINOv2FeatureExtractor()
        self.lie_reg = None
        if lie_weight > 0:
            from models.lie_group import LieRegularization
            self.lie_reg = LieRegularization(smooth_weight=lie_smooth_weight, identity_weight=lie_identity_weight, s3_weight=lie_s3_weight, temporal_weight=lie_temporal_weight)

    def forward(self, model_output: dict[str, torch.Tensor], gt: torch.Tensor) -> dict[str, torch.Tensor]:
        pred = model_output['output']
        losses = {}
        losses['pixel'] = charbonnier_loss(pred, gt, self.charbonnier_eps)
        losses['perc'] = perceptual_loss(self.feat_extractor, pred, gt)
        grids = model_output.get('grids', {})
        if self.lie_reg is not None and 'sc_grid_r' in grids:
            losses['lie'] = self.lie_reg(grids)
        else:
            losses['lie'] = torch.tensor(0.0, device=pred.device)
        losses['total'] = self.pixel_weight * losses['pixel'] + self.perc_weight * losses['perc'] + self.lie_weight * losses['lie']
        return losses
