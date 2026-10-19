import argparse
import csv
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import calc_psnr_torch
MODEL_REGISTRY: dict[str, tuple[str, str]] = {'ours': ('models', 'BilateralGridDehazeNet')}

def build_model(cfg: dict, model_name: str, checkpoint: str, device: torch.device) -> nn.Module:
    (module_path, class_name) = MODEL_REGISTRY[model_name]
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    model_cfg = cfg['model']
    model = model_class(backbone_channels=model_cfg.get('backbone_channels', [24, 48, 96, 192]), color_bins=model_cfg.get('color_bins', 8), coeff_dim=model_cfg.get('coeff_dim', 12), num_frames=model_cfg.get('temporal_num_frames', 5), encode_size=model_cfg.get('encode_size', 256), detail_unshuffle=model_cfg.get('detail_unshuffle', 4), detail_channels=model_cfg.get('detail_channels', 64), drop_path_rate=model_cfg.get('drop_path_rate', 0.1), grid_dropout=model_cfg.get('grid_dropout', 0.1), temporal_attn_heads=model_cfg.get('temporal_attn_heads', 4))
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint_data['model'] if 'model' in checkpoint_data else checkpoint_data
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Failed to load image: {path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img.copy()).permute(2, 0, 1)

def load_and_adapt_flow(flow_path: Path, target_h: int, target_w: int, device: torch.device) -> torch.Tensor:
    flow_np = np.load(str(flow_path))
    flow = torch.from_numpy(flow_np).float().to(device)
    (h_f, w_f) = flow.shape[:2]
    if h_f != target_h or w_f != target_w:
        scale_x = target_w / w_f
        scale_y = target_h / h_f
        flow_t = flow.permute(2, 0, 1).unsqueeze(0)
        flow_t = F.interpolate(flow_t, size=(target_h, target_w), mode='bilinear', align_corners=False)
        flow = flow_t.squeeze(0).permute(1, 2, 0)
        flow[..., 0] *= scale_x
        flow[..., 1] *= scale_y
    return flow

def backward_warp(img_next: torch.Tensor, flow_fwd: torch.Tensor) -> torch.Tensor:
    (H, W) = flow_fwd.shape[:2]
    device = img_next.device
    (grid_y, grid_x) = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32), torch.arange(W, device=device, dtype=torch.float32), indexing='ij')
    sample_x = (grid_x + flow_fwd[..., 0]) / (W - 1) * 2 - 1
    sample_y = (grid_y + flow_fwd[..., 1]) / (H - 1) * 2 - 1
    grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0)
    return F.grid_sample(img_next, grid, mode='bilinear', padding_mode='border', align_corners=True)

def resolve_split_dir(root: Path, split: str) -> Path:
    for name in (split, split.capitalize(), split.lower(), split.upper()):
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Split "{split}" not found under {root}')

def scan_clips(split_dir: Path) -> dict[str, dict]:
    hazy_root = split_dir / 'hazy'
    gt_root = split_dir / 'gt'
    flow_root = split_dir / 'flow'
    clips = {}
    for video_dir in sorted(hazy_root.iterdir()):
        if not video_dir.is_dir():
            continue
        name = video_dir.name
        if not (gt_root / name).exists():
            continue
        frames = sorted((p for p in video_dir.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')))
        if not frames:
            continue
        flow_dir = flow_root / name if (flow_root / name).exists() else None
        clips[name] = {'hazy_dir': video_dir, 'gt_dir': gt_root / name, 'flow_dir': flow_dir, 'n_frames': len(frames), 'ext': frames[0].suffix}
    return clips

@torch.no_grad()
def evaluate_clip(model: nn.Module, clip_info: dict, num_frames: int, device: torch.device) -> dict[str, float]:
    n_frames_total = clip_info['n_frames']
    ext = clip_info['ext']
    hazy_dir = clip_info['hazy_dir']
    gt_dir = clip_info['gt_dir']
    flow_dir = clip_info['flow_dir']
    half_t = num_frames // 2
    dehazed_cpu: dict[int, torch.Tensor] = {}
    gt_cpu: dict[int, torch.Tensor] = {}
    first_img = load_image(hazy_dir / f'{half_t:05d}{ext}')
    (target_h, target_w) = first_img.shape[:2]
    for center_idx in range(half_t, n_frames_total - half_t):
        hazy_frames = []
        for t in range(center_idx - half_t, center_idx + half_t + 1):
            img = load_image(hazy_dir / f'{t:05d}{ext}')
            hazy_frames.append(image_to_tensor(img))
        hazy_tensor = torch.stack(hazy_frames).unsqueeze(0).to(device)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            output = model.inference(hazy_tensor, target_h=target_h, target_w=target_w)
        dehazed_cpu[center_idx] = output[0].float().cpu()
        gt_img = load_image(gt_dir / f'{center_idx:05d}{ext}')
        gt_cpu[center_idx] = image_to_tensor(gt_img)
    valid_indices = sorted(dehazed_cpu.keys())
    if not valid_indices:
        return {'tOF': None, 'sigma_psnr': 0.0, 'mean_psnr': 0.0, 'n_valid': 0, 'n_pairs': 0}
    psnrs = []
    for idx in valid_indices:
        psnrs.append(calc_psnr_torch(dehazed_cpu[idx], gt_cpu[idx]))
    psnrs_arr = np.array(psnrs, dtype=np.float64)
    mean_psnr = float(psnrs_arr.mean())
    sigma_psnr = float(psnrs_arr.std())
    tof_values: list[float] = []
    if flow_dir is not None:
        for i in range(len(valid_indices) - 1):
            idx_t = valid_indices[i]
            idx_t1 = valid_indices[i + 1]
            if idx_t1 != idx_t + 1:
                continue
            flow_path = flow_dir / f'{idx_t:05d}.npy'
            if not flow_path.exists():
                continue
            (out_h, out_w) = (dehazed_cpu[idx_t].shape[1], dehazed_cpu[idx_t].shape[2])
            flow = load_and_adapt_flow(flow_path, out_h, out_w, device)
            img_t = dehazed_cpu[idx_t].unsqueeze(0).to(device)
            img_t1 = dehazed_cpu[idx_t1].unsqueeze(0).to(device)
            warped_t1 = backward_warp(img_t1, flow)
            l2_err = ((img_t - warped_t1) ** 2).mean().item()
            tof_values.append(l2_err)
    tof_mean = float(np.mean(tof_values)) if tof_values else None
    return {'tOF': tof_mean, 'sigma_psnr': sigma_psnr, 'mean_psnr': mean_psnr, 'n_valid': len(valid_indices), 'n_pairs': len(tof_values)}

def evaluate_model(cfg: dict, model_name: str, checkpoint: str, output_dir: Path, device: torch.device) -> dict:
    print(f"\n{'=' * 60}")
    print(f'Evaluating: {model_name}')
    print(f'Checkpoint: {checkpoint}')
    print(f"{'=' * 60}")
    model = build_model(cfg, model_name, checkpoint, device)
    n_params = sum((p.numel() for p in model.parameters())) / 1000000.0
    print(f'  Loaded — {n_params:.2f}M params')
    data_cfg = cfg['data']
    split_name = data_cfg.get('test_split', 'Test')
    dataset_root = Path(data_cfg['dataset_root'])
    split_dir = resolve_split_dir(dataset_root, split_name)
    num_frames = data_cfg['num_frames']
    clips = scan_clips(split_dir)
    if not clips:
        print(f'  [WARN] No clips found in {split_dir}')
        return {}
    print(f'  Found {len(clips)} clips, num_frames={num_frames}')
    per_clip_results = {}
    all_tof: list[float] = []
    all_sigma_psnr: list[float] = []
    all_mean_psnr: list[float] = []
    total_pairs = 0
    for clip_name in tqdm(sorted(clips.keys()), desc=f'  {model_name}', leave=False):
        clip_info = clips[clip_name]
        result = evaluate_clip(model, clip_info, num_frames, device)
        per_clip_results[clip_name] = result
        if result['tOF'] is not None:
            all_tof.append(result['tOF'])
        all_sigma_psnr.append(result['sigma_psnr'])
        all_mean_psnr.append(result['mean_psnr'])
        total_pairs += result['n_pairs']
    overall = {'tOF': float(np.mean(all_tof)) if all_tof else None, 'sigma_psnr': float(np.mean(all_sigma_psnr)) if all_sigma_psnr else None, 'mean_psnr': float(np.mean(all_mean_psnr)) if all_mean_psnr else None, 'total_clips': len(clips), 'total_pairs': total_pairs}
    record = {'model': model_name, 'checkpoint': checkpoint, 'dataset': str(dataset_root), 'timestamp': datetime.now().isoformat(timespec='seconds'), 'metrics': overall, 'per_clip': per_clip_results}
    per_clip_dir = output_dir / 'per_clip'
    per_clip_dir.mkdir(parents=True, exist_ok=True)
    json_path = per_clip_dir / f'{model_name}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    tof_str = f"{overall['tOF']:.6f}" if overall['tOF'] is not None else 'N/A'
    sig_str = f"{overall['sigma_psnr']:.4f}" if overall['sigma_psnr'] is not None else 'N/A'
    psnr_str = f"{overall['mean_psnr']:.4f}" if overall['mean_psnr'] is not None else 'N/A'
    print(f"  Results: tOF={tof_str}  sigma-PSNR={sig_str}  mean-PSNR={psnr_str}  ({overall['total_clips']} clips, {total_pairs} pairs)")
    del model
    torch.cuda.empty_cache()
    return record

def save_summary(records: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_data = []
    for rec in records:
        summary_data.append({'model': rec['model'], 'checkpoint': rec['checkpoint'], **{k: v for (k, v) in rec['metrics'].items()}})
    json_path = output_dir / 'summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    csv_path = output_dir / 'summary.csv'
    fieldnames = ['model', 'tOF', 'sigma_psnr', 'mean_psnr', 'total_clips', 'total_pairs']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in summary_data:
            writer.writerow({k: item.get(k, '') for k in fieldnames})
    print(f"\n{'=' * 80}")
    print(f"{'Model':<20} {'tOF':>10} {'σ-PSNR':>10} {'PSNR':>10} {'Clips':>6} {'Pairs':>6}")
    print(f"{'-' * 80}")
    for item in summary_data:
        tof = f"{item['tOF']:.6f}" if item.get('tOF') is not None else 'N/A'
        sig = f"{item['sigma_psnr']:.4f}" if item.get('sigma_psnr') is not None else 'N/A'
        psnr = f"{item['mean_psnr']:.4f}" if item.get('mean_psnr') is not None else 'N/A'
        print(f"{item['model']:<20} {tof:>10} {sig:>10} {psnr:>10} {item.get('total_clips', ''):>6} {item.get('total_pairs', ''):>6}")
    print(f"{'=' * 80}")
    print(f'\nSummary saved → {json_path}')
    print(f'Summary saved → {csv_path}')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Temporal consistency evaluation (tOF / sigma-PSNR)', formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('--config', type=str, required=True, help='Dataset config yaml')
    parser.add_argument('--model', type=str, default=None, choices=list(MODEL_REGISTRY.keys()), help='Single model to evaluate')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path (single model mode)')
    parser.add_argument('--all_models', action='store_true', help='Evaluate all models in registry')
    parser.add_argument('--checkpoint_root', type=str, default=None, help='Root dir for checkpoints (batch mode): {root}/{model_name}/best|last.pth')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(cfg['test']['output_dir']).parent / 'temporal_consistency'
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    if args.all_models:
        assert args.checkpoint_root is not None, '--checkpoint_root required with --all_models'
        ckpt_root = Path(args.checkpoint_root)
        for model_name in MODEL_REGISTRY:
            if model_name == 'ours':
                ckpt = ckpt_root / 'ours' / 'best.pth'
            else:
                ckpt = ckpt_root / model_name / 'last.pth'
            if not ckpt.exists():
                print(f'[SKIP] {model_name}: {ckpt} not found')
                continue
            rec = evaluate_model(cfg, model_name, str(ckpt), output_dir, device)
            if rec:
                records.append(rec)
    elif args.model is not None:
        assert args.checkpoint is not None, '--checkpoint required with --model'
        rec = evaluate_model(cfg, args.model, args.checkpoint, output_dir, device)
        if rec:
            records.append(rec)
    else:
        print('Error: specify --model + --checkpoint, or --all_models + --checkpoint_root')
        sys.exit(1)
    if records:
        save_summary(records, output_dir)
if __name__ == '__main__':
    main()
