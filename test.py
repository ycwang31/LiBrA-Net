import argparse
import importlib
import json
import logging
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from data import VideoDehazeDataset
from utils import build_lpips_metric, evaluate_dehazing_model
logger = logging.getLogger('dehazing')
MODEL_REGISTRY: dict[str, tuple[str, str]] = {'ours': ('models', 'BilateralGridDehazeNet')}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Video Dehazing Testing')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--model', type=str, default='ours', choices=list(MODEL_REGISTRY.keys()), help='Model name')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory for saved results')
    parser.add_argument('--save_images', action='store_true', help='Save dehazed output images')
    parser.add_argument('--flops_size', type=str, default='3840x2160', help='HxW resolution for FLOPs computation')
    return parser.parse_args()

def load_config(path: str) -> dict:
    with open(path) as file:
        return yaml.safe_load(file)

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

def build_test_loader(cfg: dict) -> DataLoader:
    data_cfg = cfg['data']
    split_name = data_cfg.get('test_split', 'Test')
    dataset = VideoDehazeDataset(root=data_cfg['dataset_root'], split=split_name, num_frames=data_cfg['num_frames'], crop_size=0)
    num_workers = int(data_cfg.get('test_num_workers', data_cfg.get('num_workers', 0)))
    loader_kwargs = {'dataset': dataset, 'batch_size': 1, 'shuffle': False, 'num_workers': num_workers, 'pin_memory': data_cfg.get('pin_memory', True), 'drop_last': False}
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = data_cfg.get('persistent_workers', False)
        if data_cfg.get('prefetch_factor') is not None:
            loader_kwargs['prefetch_factor'] = data_cfg['prefetch_factor']
    return DataLoader(**loader_kwargs)

class _InferenceWrapper(nn.Module):

    def __init__(self, model: nn.Module, h: int, w: int) -> None:
        super().__init__()
        self.model = model
        self.h = h
        self.w = w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.inference(x, target_h=self.h, target_w=self.w)

def _strip_thop_residue_from_module(module: nn.Module) -> None:
    for m in module.modules():
        m._forward_hooks.clear()
        m._forward_hooks_with_kwargs.clear()
        m._forward_hooks_always_called.clear()
        for k in ('total_ops', 'total_params'):
            m._buffers.pop(k, None)

def compute_flops(model: nn.Module, num_frames: int, height: int, width: int, device: torch.device) -> float | None:
    try:
        from thop import profile
    except ImportError:
        print('[WARN] thop not installed, skipping FLOPs. Install: pip install thop')
        return None
    wrapper = _InferenceWrapper(model, height, width).to(device).eval()
    dummy = torch.randn(1, num_frames, 3, height, width, device=device, dtype=torch.float16)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    macs: float | None = None
    try:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            (macs, _) = profile(wrapper, inputs=(dummy,), verbose=False)
    except Exception as e:
        print(f'[WARN] FLOPs computation failed at {height}x{width}: {e}')
        return None
    finally:
        _strip_thop_residue_from_module(wrapper)
        del wrapper, dummy
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if macs is None:
        return None
    return macs * 2 / 1000000000.0

def format_results(results: dict[str, float | int | None]) -> str:
    lines = ['\n===== Test Results =====']
    if results.get('params_m') is not None:
        lines.append(f"Params:     {results['params_m']:.2f}M")
    if results.get('gflops') is not None:
        lines.append(f"GFLOPs:     {results['gflops']:.2f}  ({results.get('flops_size', 'N/A')})")
    if results['psnr'] is not None:
        lines.append(f"PSNR:       {results['psnr']:.4f} dB")
    if results['ssim'] is not None:
        lines.append(f"SSIM:       {results['ssim']:.6f}")
    if results['lpips'] is not None:
        lines.append(f"LPIPS:      {results['lpips']:.6f}")
    if results['fps'] is not None:
        lines.append(f"FPS:        {results['fps']:.2f}")
    if results['ms_per_frame'] is not None:
        lines.append(f"ms/frame:   {results['ms_per_frame']:.2f}")
    lines.append(f"Total frames: {results['total_frames']}")
    return '\n'.join(lines)

@torch.no_grad()
def test(cfg: dict, args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg, args.model, args.checkpoint, device)
    n_params = sum((p.numel() for p in model.parameters())) / 1000000.0
    print(f'Model [{args.model}] loaded — {n_params:.2f}M params')
    (flops_h, flops_w) = (int(x) for x in args.flops_size.split('x'))
    num_frames = cfg['data']['num_frames']
    gflops = compute_flops(model, num_frames, flops_h, flops_w, device)
    if gflops is not None:
        print(f'GFLOPs: {gflops:.2f}  (input {args.flops_size}, T={num_frames})')
    test_loader = build_test_loader(cfg)
    eval_cfg = cfg.get('eval', {})
    metric_names = [metric.lower() for metric in eval_cfg.get('metrics', ['psnr', 'ssim', 'lpips', 'fps'])]
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(cfg['test']['output_dir']) / args.model
    save_dir = output_dir if args.save_images or cfg['test'].get('save_images', False) else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    lpips_fn = build_lpips_metric(device) if 'lpips' in metric_names else None
    results = evaluate_dehazing_model(model=model, data_loader=test_loader, device=device, metric_names=metric_names, distributed=False, lpips_fn=lpips_fn, save_dir=save_dir, show_progress=True, description='Testing', amp_inference=True, gpu_metrics=True)
    results['params_m'] = n_params
    if gflops is not None:
        results['gflops'] = gflops
        results['flops_size'] = f'{args.flops_size}, T={num_frames}'
    result_text = format_results(results)
    print(result_text)
    output_dir.mkdir(parents=True, exist_ok=True)
    encode_size = cfg['model'].get('encode_size', 256)
    metrics_record = {'model': args.model, 'checkpoint': args.checkpoint, 'config': args.config, 'encode_size': encode_size, 'timestamp': datetime.now().isoformat(timespec='seconds'), 'results': {k: v for (k, v) in results.items() if v is not None}}
    json_path = output_dir / 'metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_record, f, indent=2, ensure_ascii=False)
    txt_path = output_dir / 'metrics.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f'Model: {args.model}\n')
        f.write(f'Checkpoint: {args.checkpoint}\n')
        f.write(f'Config: {args.config}\n')
        f.write(f"Timestamp: {metrics_record['timestamp']}\n")
        f.write(result_text + '\n')
    print(f'\nMetrics saved → {json_path}')
    print(f'Metrics saved → {txt_path}')
if __name__ == '__main__':
    parsed_args = parse_args()
    config = load_config(parsed_args.config)
    test(config, parsed_args)
