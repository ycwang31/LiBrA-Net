import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm.auto import tqdm
MODEL_REGISTRY: dict[str, tuple[str, str]] = {'ours': ('models', 'BilateralGridDehazeNet')}
NR_METRIC_INFO: dict[str, tuple[str, bool]] = {'musiq': ('musiq', False), 'clipiqa': ('clipiqa+', False), 'brisque': ('brisque', True), 'niqe': ('niqe', True)}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Real-world Video Dehazing — NR-IQA Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Model config YAML')
    parser.add_argument('--model', type=str, default='ours', choices=list(MODEL_REGISTRY.keys()), help='Model name')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory with clip subdirs (01/, 02/, ...)')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--save_images', action='store_true', help='Save dehazed images')
    parser.add_argument('--samples_per_clip', type=int, default=0, help='Frames to sample per clip (0 = all valid frames)')
    parser.add_argument('--metrics', type=str, nargs='+', default=['musiq', 'brisque', 'niqe'], choices=list(NR_METRIC_INFO.keys()), help='NR-IQA metrics to compute')
    return parser.parse_args()

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def build_model(cfg: dict, model_name: str, checkpoint: str, device: torch.device) -> nn.Module:
    (module_path, class_name) = MODEL_REGISTRY[model_name]
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    model_cfg = cfg['model']
    model = model_class(backbone_channels=model_cfg.get('backbone_channels', [24, 48, 96, 192]), color_bins=model_cfg.get('color_bins', 8), coeff_dim=model_cfg.get('coeff_dim', 12), num_frames=model_cfg.get('temporal_num_frames', 5), encode_size=model_cfg.get('encode_size', 256), detail_unshuffle=model_cfg.get('detail_unshuffle', 4), detail_channels=model_cfg.get('detail_channels', 64), drop_path_rate=model_cfg.get('drop_path_rate', 0.1), grid_dropout=model_cfg.get('grid_dropout', 0.1), temporal_attn_heads=model_cfg.get('temporal_attn_heads', 4))
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def scan_clips(video_dir: Path, num_frames: int, samples_per_clip: int) -> list[dict]:
    half_t = num_frames // 2
    clips = []
    for clip_dir in sorted(video_dir.iterdir()):
        if not clip_dir.is_dir():
            continue
        frames = sorted((p for p in clip_dir.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')))
        n_frames = len(frames)
        if n_frames < num_frames:
            print(f'[WARN] Clip {clip_dir.name}: {n_frames} frames < {num_frames} required, skipping.')
            continue
        valid_start = half_t
        valid_end = n_frames - half_t - 1
        n_valid = valid_end - valid_start + 1
        if samples_per_clip <= 0 or samples_per_clip >= n_valid:
            indices = list(range(valid_start, valid_end + 1))
        elif samples_per_clip == 1:
            indices = [(valid_start + valid_end) // 2]
        else:
            indices = [valid_start + round(i * (n_valid - 1) / (samples_per_clip - 1)) for i in range(samples_per_clip)]
        clips.append({'name': clip_dir.name, 'dir': clip_dir, 'center_indices': indices, 'n_frames': n_frames, 'ext': frames[0].suffix})
    return clips

def load_window(clip_dir: Path, center_idx: int, half_t: int, ext: str, device: torch.device) -> torch.Tensor:
    frames = []
    for t in range(center_idx - half_t, center_idx + half_t + 1):
        path = clip_dir / f'{t:05d}{ext}'
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        frames.append(tensor)
    return torch.stack(frames).unsqueeze(0).to(device)

def build_nr_metrics(metric_names: list[str], device: torch.device) -> dict[str, nn.Module]:
    try:
        import pyiqa
    except ImportError:
        raise ImportError('pyiqa is required for NR-IQA metrics. Install: pip install pyiqa') from None
    metrics = {}
    for name in metric_names:
        pyiqa_name = NR_METRIC_INFO[name][0]
        print(f'  Loading {name} ({pyiqa_name}) ...')
        metrics[name] = pyiqa.create_metric(pyiqa_name, device=device)
    return metrics

@torch.no_grad()
def compute_nr_scores(img: torch.Tensor, metrics: dict[str, nn.Module]) -> dict[str, float]:
    scores = {}
    for (name, metric_fn) in metrics.items():
        try:
            scores[name] = float(metric_fn(img).item())
        except Exception as e:
            print(f'  [WARN] {name} failed: {e}')
            scores[name] = float('nan')
    return scores

def _delta_str(hazy: float, dehazed: float, lower_is_better: bool) -> str:
    if abs(hazy) < 1e-08:
        return '  N/A'
    pct = (dehazed - hazy) / abs(hazy) * 100
    improved = pct < 0 if lower_is_better else pct > 0
    arrow = '✓' if improved else '✗'
    return f'{pct:+.1f}% {arrow}'

def format_results_table(clip_results: list[dict], metric_names: list[str], model_name: str) -> str:
    col_w = 17
    lines: list[str] = []
    sep = '=' * (15 + len(metric_names) * (col_w + 2))
    lines.append('')
    lines.append(sep)
    lines.append(f'Real-world NR-IQA Results — Model: {model_name}')
    lines.append(sep)
    hdr = f"{'Clip':>6s} {'Sampled':>8s}"
    for m in metric_names:
        arrow = '↓' if NR_METRIC_INFO[m][1] else '↑'
        label = f'{m.upper()}{arrow}'
        hdr += f'  {label:>{col_w}s}'
    lines.append(hdr)
    sub = f"{'':>6s} {'':>8s}"
    for _ in metric_names:
        sub += f"  {'hazy → dehazed':>{col_w}s}"
    lines.append(sub)
    lines.append('-' * len(sep))
    for r in clip_results:
        row = f"{r['clip']:>6s} {r['n_sampled']:>4d}/{r['n_frames']:<4d}"
        for m in metric_names:
            (h, d) = (r[f'hazy_{m}'], r[f'dehazed_{m}'])
            cell = f'{h:.2f} → {d:.2f}'
            row += f'  {cell:>{col_w}s}'
        lines.append(row)
    lines.append('-' * len(sep))
    total_sampled = sum((r['n_sampled'] for r in clip_results))
    mean_row = f"{'Mean':>6s} {total_sampled:>8d}"
    for m in metric_names:
        h_avg = float(np.nanmean([r[f'hazy_{m}'] for r in clip_results]))
        d_avg = float(np.nanmean([r[f'dehazed_{m}'] for r in clip_results]))
        cell = f'{h_avg:.2f} → {d_avg:.2f}'
        mean_row += f'  {cell:>{col_w}s}'
    lines.append(mean_row)
    delta_row = f"{'Delta':>6s} {'':>8s}"
    for m in metric_names:
        h_avg = float(np.nanmean([r[f'hazy_{m}'] for r in clip_results]))
        d_avg = float(np.nanmean([r[f'dehazed_{m}'] for r in clip_results]))
        cell = _delta_str(h_avg, d_avg, NR_METRIC_INFO[m][1])
        delta_row += f'  {cell:>{col_w}s}'
    lines.append(delta_row)
    lines.append(sep)
    return '\n'.join(lines)

@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg, args.model, args.checkpoint, device)
    n_params = sum((p.numel() for p in model.parameters())) / 1000000.0
    print(f'Model [{args.model}] loaded — {n_params:.2f}M params')
    print(f'Loading NR-IQA metrics: {args.metrics}')
    nr_metrics = build_nr_metrics(args.metrics, device)
    print(f'All {len(nr_metrics)} metrics ready.\n')
    video_dir = Path(args.video_dir)
    num_frames = cfg['data']['num_frames']
    clips = scan_clips(video_dir, num_frames, args.samples_per_clip)
    if not clips:
        print(f'[ERROR] No valid clips found in {video_dir}')
        return
    total_samples = sum((len(c['center_indices']) for c in clips))
    print(f'Found {len(clips)} clips, {total_samples} sampled frames\n')
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        base = Path(cfg.get('test', {}).get('output_dir', './results/4krealvideos'))
        output_dir = base / f'{args.model}_nriqa'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir = output_dir / 'dehazed' if args.save_images else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    half_t = num_frames // 2
    clip_results: list[dict] = []
    pbar = tqdm(total=total_samples, desc='NR-IQA Evaluation')
    for clip in clips:
        clip_hazy: dict[str, list[float]] = {m: [] for m in args.metrics}
        clip_dehazed: dict[str, list[float]] = {m: [] for m in args.metrics}
        for center_idx in clip['center_indices']:
            hazy_window = load_window(clip['dir'], center_idx, half_t, clip['ext'], device)
            hazy_center = hazy_window[:, half_t]
            (_, _, target_h, target_w) = hazy_center.shape
            with torch.amp.autocast('cuda', dtype=torch.float16):
                dehazed = model.inference(hazy_window, target_h=target_h, target_w=target_w)
            dehazed = dehazed.float().clamp(0, 1)
            hazy_scores = compute_nr_scores(hazy_center, nr_metrics)
            dehazed_scores = compute_nr_scores(dehazed, nr_metrics)
            for m in args.metrics:
                clip_hazy[m].append(hazy_scores[m])
                clip_dehazed[m].append(dehazed_scores[m])
            if save_dir:
                clip_save_dir = save_dir / clip['name']
                clip_save_dir.mkdir(parents=True, exist_ok=True)
                img_np = (dehazed[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                cv2.imwrite(str(clip_save_dir / f'{center_idx:05d}.png'), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            del hazy_window, hazy_center, dehazed
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            pbar.update(1)
        result = {'clip': clip['name'], 'n_frames': clip['n_frames'], 'n_sampled': len(clip['center_indices'])}
        for m in args.metrics:
            result[f'hazy_{m}'] = float(np.nanmean(clip_hazy[m]))
            result[f'dehazed_{m}'] = float(np.nanmean(clip_dehazed[m]))
        clip_results.append(result)
    pbar.close()
    table = format_results_table(clip_results, args.metrics, args.model)
    print(table)
    summary: dict = {'model': args.model, 'checkpoint': args.checkpoint, 'config': args.config, 'video_dir': str(video_dir), 'samples_per_clip': args.samples_per_clip, 'metrics': args.metrics, 'timestamp': datetime.now().isoformat(timespec='seconds'), 'overall': {}, 'per_clip': clip_results}
    for m in args.metrics:
        h_avg = float(np.nanmean([r[f'hazy_{m}'] for r in clip_results]))
        d_avg = float(np.nanmean([r[f'dehazed_{m}'] for r in clip_results]))
        summary['overall'][f'hazy_{m}'] = h_avg
        summary['overall'][f'dehazed_{m}'] = d_avg
    json_path = output_dir / 'nriqa_metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    txt_path = output_dir / 'nriqa_metrics.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f'Model: {args.model}\n')
        f.write(f'Checkpoint: {args.checkpoint}\n')
        f.write(f'Video dir: {video_dir}\n')
        f.write(f'Samples per clip: {args.samples_per_clip}\n')
        f.write(f"Timestamp: {summary['timestamp']}\n")
        f.write(table + '\n')
    print(f'\nMetrics saved → {json_path}')
    print(f'Metrics saved → {txt_path}')
    if save_dir:
        print(f'Dehazed images → {save_dir}')
if __name__ == '__main__':
    main()
