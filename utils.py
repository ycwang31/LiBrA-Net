import logging
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from tqdm.auto import tqdm

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_distributed(backend: str='nccl', timeout_minutes: int=180) -> tuple[int, int, int]:
    if dist.is_initialized():
        return (dist.get_rank(), dist.get_world_size(), int(os.environ.get('LOCAL_RANK', 0)))
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_minutes))
        return (rank, world_size, local_rank)
    return (0, 1, 0)

def is_main_process() -> bool:
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True

def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return tensor
    reduced = tensor.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist.get_world_size()
    return reduced

def reduce_tensor_dict(tensor_dict: dict[str, torch.Tensor], average: bool=True) -> dict[str, torch.Tensor]:
    if not dist.is_initialized():
        return tensor_dict
    keys = list(tensor_dict.keys())
    values = torch.stack([tensor_dict[key].detach() for key in keys], dim=0)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    if average:
        values /= dist.get_world_size()
    return {key: values[idx] for (idx, key) in enumerate(keys)}

def all_reduce_scalar_dict(scalar_dict: dict[str, float], device: torch.device, average: bool=False) -> dict[str, float]:
    if not scalar_dict:
        return {}
    if not dist.is_initialized():
        return scalar_dict
    keys = list(scalar_dict.keys())
    values = torch.tensor([scalar_dict[key] for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    if average:
        values /= dist.get_world_size()
    return {key: values[idx].item() for (idx, key) in enumerate(keys)}

def calc_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(compare_psnr(gt, pred, data_range=255))

def calc_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(compare_ssim(gt, pred, channel_axis=2, data_range=255))

def tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dim() == 3 and tensor.shape[0] in (1, 3):
        tensor = tensor.permute(1, 2, 0)
    img = tensor.detach().cpu().clamp(0, 1).numpy()
    return (img * 255).astype(np.uint8)

def _uniform_kernel_2d(window_size: int=7) -> torch.Tensor:
    return torch.ones(window_size, window_size, dtype=torch.float32) / window_size ** 2

def calc_psnr_torch(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred_q = (pred.clamp(0, 1) * 255).round()
    gt_q = (gt.clamp(0, 1) * 255).round()
    mse = ((pred_q - gt_q) ** 2).float().mean()
    if mse < 1e-10:
        return 100.0
    return float(10.0 * torch.log10(255.0 ** 2 / mse))

def calc_ssim_torch(pred: torch.Tensor, gt: torch.Tensor, window_size: int=7) -> float:
    pred_q = (pred.clamp(0, 1) * 255).round().float().unsqueeze(0)
    gt_q = (gt.clamp(0, 1) * 255).round().float().unsqueeze(0)
    C_dim = pred_q.shape[1]
    kernel = _uniform_kernel_2d(window_size).to(pred.device, dtype=pred_q.dtype)
    kernel = kernel.expand(C_dim, 1, window_size, window_size)
    pad = window_size // 2
    mu1 = F.conv2d(pred_q, kernel, padding=pad, groups=C_dim)
    mu2 = F.conv2d(gt_q, kernel, padding=pad, groups=C_dim)
    (mu1_sq, mu2_sq, mu12) = (mu1 * mu1, mu2 * mu2, mu1 * mu2)
    sigma1_sq = F.conv2d(pred_q * pred_q, kernel, padding=pad, groups=C_dim) - mu1_sq
    sigma2_sq = F.conv2d(gt_q * gt_q, kernel, padding=pad, groups=C_dim) - mu2_sq
    sigma12 = F.conv2d(pred_q * gt_q, kernel, padding=pad, groups=C_dim) - mu12
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    ssim_map = (2 * mu12 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_map = ssim_map[:, :, pad:-pad, pad:-pad]
    return float(ssim_map.mean())

def build_lpips_metric(device: torch.device, logger: logging.Logger | None=None) -> nn.Module | None:
    try:
        import lpips
    except ImportError:
        if logger is not None:
            logger.warning('lpips is not installed; LPIPS metric will be skipped.')
        return None
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()
    return lpips_fn

@torch.no_grad()
def evaluate_dehazing_model(model: nn.Module, data_loader, device: torch.device, metric_names: list[str], distributed: bool=False, lpips_fn: nn.Module | None=None, save_dir: Path | None=None, show_progress: bool=False, description: str='Evaluation', amp_inference: bool=False, gpu_metrics: bool=False, warmup_batches: int=5) -> dict[str, float | int | None]:
    raw_model = model.module if hasattr(model, 'module') else model
    metric_set = {name.lower() for name in metric_names}
    use_timing = 'fps' in metric_set
    use_cuda_timing = use_timing and device.type == 'cuda'
    metric_sums = {'psnr_sum': 0.0, 'ssim_sum': 0.0, 'lpips_sum': 0.0, 'elapsed_ms_sum': 0.0, 'frame_count': 0.0, 'timing_frame_count': 0.0, 'lpips_count': 0.0}
    iterator = data_loader
    if show_progress:
        iterator = tqdm(data_loader, desc=description, leave=False)
    batch_idx = 0
    for batch in iterator:
        hazy_frames = batch['hazy_frames'].to(device, non_blocking=True)
        clear = batch['clear'].to(device, non_blocking=True)
        (_, _, _, target_h, target_w) = hazy_frames.shape
        is_warmup = use_timing and batch_idx < warmup_batches
        if use_cuda_timing:
            torch.cuda.synchronize(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        elif use_timing:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=amp_inference):
            output = raw_model.inference(hazy_frames, target_h=target_h, target_w=target_w)
        elapsed_ms = 0.0
        if use_cuda_timing:
            end_event.record()
            torch.cuda.synchronize(device)
            elapsed_ms = float(start_event.elapsed_time(end_event))
        elif use_timing:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        batch_size = int(output.shape[0])
        metric_sums['frame_count'] += batch_size
        if not is_warmup:
            metric_sums['elapsed_ms_sum'] += elapsed_ms
            metric_sums['timing_frame_count'] += batch_size
        if 'lpips' in metric_set and lpips_fn is not None:
            lpips_values = lpips_fn(output * 2 - 1, clear * 2 - 1).view(-1)
            metric_sums['lpips_sum'] += float(lpips_values.sum().item())
            metric_sums['lpips_count'] += float(lpips_values.numel())
        for idx in range(batch_size):
            if gpu_metrics:
                if 'psnr' in metric_set:
                    metric_sums['psnr_sum'] += calc_psnr_torch(output[idx], clear[idx])
                if 'ssim' in metric_set:
                    metric_sums['ssim_sum'] += calc_ssim_torch(output[idx], clear[idx])
            else:
                pred_np = tensor_to_uint8(output[idx])
                gt_np = tensor_to_uint8(clear[idx])
                if 'psnr' in metric_set:
                    metric_sums['psnr_sum'] += calc_psnr(pred_np, gt_np)
                if 'ssim' in metric_set:
                    metric_sums['ssim_sum'] += calc_ssim(pred_np, gt_np)
            if save_dir is not None:
                pred_np_save = tensor_to_uint8(output[idx]) if gpu_metrics else pred_np
                video_name = batch['video_name'][idx]
                frame_idx = int(batch['frame_idx'][idx])
                video_dir = save_dir / video_name
                video_dir.mkdir(parents=True, exist_ok=True)
                save_path = video_dir / f'{frame_idx:05d}.png'
                pred_bgr = cv2.cvtColor(pred_np_save, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(save_path), pred_bgr)
        batch_idx += 1
    if distributed and dist.is_initialized():
        reduced = torch.tensor([metric_sums['psnr_sum'], metric_sums['ssim_sum'], metric_sums['lpips_sum'], metric_sums['elapsed_ms_sum'], metric_sums['frame_count'], metric_sums['timing_frame_count'], metric_sums['lpips_count']], device=device, dtype=torch.float64)
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        metric_sums = {'psnr_sum': reduced[0].item(), 'ssim_sum': reduced[1].item(), 'lpips_sum': reduced[2].item(), 'elapsed_ms_sum': reduced[3].item(), 'frame_count': reduced[4].item(), 'timing_frame_count': reduced[5].item(), 'lpips_count': reduced[6].item()}
    frame_count = max(int(metric_sums['frame_count']), 0)
    timing_frame_count = max(int(metric_sums['timing_frame_count']), 0)
    lpips_count = max(int(metric_sums['lpips_count']), 0)
    elapsed_ms_sum = metric_sums['elapsed_ms_sum']
    results: dict[str, float | int | None] = {'total_frames': frame_count, 'psnr': None, 'ssim': None, 'lpips': None, 'fps': None, 'ms_per_frame': None}
    if frame_count > 0:
        if 'psnr' in metric_set:
            results['psnr'] = metric_sums['psnr_sum'] / frame_count
        if 'ssim' in metric_set:
            results['ssim'] = metric_sums['ssim_sum'] / frame_count
        if use_timing and elapsed_ms_sum > 0 and (timing_frame_count > 0):
            results['fps'] = timing_frame_count * 1000.0 / elapsed_ms_sum
            results['ms_per_frame'] = elapsed_ms_sum / timing_frame_count
    if 'lpips' in metric_set and lpips_count > 0:
        results['lpips'] = metric_sums['lpips_sum'] / lpips_count
    return results

def get_cosine_schedule_with_warmup(optimizer: torch.optim.Optimizer, warmup_iters: int, total_iters: int) -> torch.optim.lr_scheduler.LambdaLR:

    def lr_lambda(current_iter: int) -> float:
        if current_iter < warmup_iters:
            return current_iter / max(1, warmup_iters)
        progress = (current_iter - warmup_iters) / max(1, total_iters - warmup_iters)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def setup_logger(name: str='dehazing', log_dir: str | Path | None=None, log_filename: str='train.log', level: int=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_dir / log_filename), encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

class AverageMeter:

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)
