from __future__ import annotations
import math
from typing import Any
import cv2
import numpy as np
METRIC_DIRECTION: dict[str, str] = {'niqe': 'down', 'brisque': 'down', 'musiq': 'up', 'det_count': 'up', 'mean_conf': 'up', 'high_conf_rate': 'up', 'small_obj_count': 'up', 'boundary_energy': 'up', 'class_entropy': 'up', 'n_components': 'up'}
METRIC_PRETTY: dict[str, str] = {'niqe': 'NIQE', 'brisque': 'BRISQUE', 'musiq': 'MUSIQ', 'det_count': 'DC', 'mean_conf': 'mConf', 'high_conf_rate': 'HCR', 'small_obj_count': 'SObj', 'boundary_energy': 'BdE', 'class_entropy': 'ClsH', 'n_components': 'NC'}
IMAGE_METRIC_KEYS: tuple[str, ...] = ('niqe', 'brisque', 'musiq')
DETECTION_METRIC_KEYS: tuple[str, ...] = ('det_count', 'mean_conf', 'high_conf_rate', 'small_obj_count')
SEGMENTATION_METRIC_KEYS: tuple[str, ...] = ('boundary_energy', 'class_entropy', 'n_components')
ALL_METRIC_KEYS: tuple[str, ...] = IMAGE_METRIC_KEYS + DETECTION_METRIC_KEYS + SEGMENTATION_METRIC_KEYS

class NRImageQualityEvaluator:

    def __init__(self, metrics: tuple[str, ...]=IMAGE_METRIC_KEYS, device: str='cuda', fp16: bool=True, max_side: int=0) -> None:
        for m in metrics:
            if m not in IMAGE_METRIC_KEYS:
                raise ValueError(f'Unknown IQA metric {m!r}; expected subset of {IMAGE_METRIC_KEYS}.')
        import torch
        try:
            import pyiqa
        except ImportError as e:
            raise ImportError('pyiqa is required for no-reference image quality metrics. Install with: pip install pyiqa>=0.1.11') from e
        self._torch = torch
        self._pyiqa = pyiqa
        cuda_ok = torch.cuda.is_available()
        self.device = device if cuda_ok or device == 'cpu' else 'cpu'
        self.fp16 = bool(fp16) and self.device.startswith('cuda')
        if max_side < 0:
            raise ValueError(f'max_side must be >= 0, got {max_side}.')
        self.max_side = int(max_side)
        self._metrics: dict[str, Any] = {}
        self._metric_devices: dict[str, str] = {}
        for name in metrics:
            try:
                metric = pyiqa.create_metric(name, device=self.device, as_loss=False)
                self._metric_devices[name] = self.device
            except Exception as exc:
                print(f'[metrics] WARN pyiqa.create_metric({name!r}) on {self.device} failed: {exc!r}. Retrying on CPU.')
                metric = pyiqa.create_metric(name, device='cpu', as_loss=False)
                self._metric_devices[name] = 'cpu'
            if hasattr(metric, 'eval'):
                metric.eval()
            self._metrics[name] = metric
        self.metric_names: tuple[str, ...] = tuple(self._metrics.keys())

    def _maybe_downsample(self, image_rgb_u8: np.ndarray) -> np.ndarray:
        if self.max_side <= 0:
            return image_rgb_u8
        (h, w) = image_rgb_u8.shape[:2]
        long_side = max(h, w)
        if long_side <= self.max_side:
            return image_rgb_u8
        scale = self.max_side / float(long_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(image_rgb_u8, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _to_tensor(self, image_rgb_u8: np.ndarray) -> Any:
        if image_rgb_u8.ndim != 3 or image_rgb_u8.shape[2] != 3:
            raise ValueError(f'Expected [H,W,3] RGB uint8, got {image_rgb_u8.shape}.')
        if image_rgb_u8.dtype != np.uint8:
            raise ValueError(f'Expected uint8 RGB, got {image_rgb_u8.dtype}.')
        torch = self._torch
        arr = image_rgb_u8.astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()

    def __call__(self, image_rgb_u8: np.ndarray) -> dict[str, float]:
        torch = self._torch
        image = self._maybe_downsample(image_rgb_u8)
        t_cpu = self._to_tensor(image)
        out: dict[str, float] = {}
        for (name, metric) in self._metrics.items():
            dev = self._metric_devices[name]
            try:
                t = t_cpu.to(dev, non_blocking=True)
                with torch.inference_mode():
                    if self.fp16 and str(dev).startswith('cuda'):
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            v = metric(t)
                    else:
                        v = metric(t)
                out[name] = float(v.detach().float().cpu().item())
            except torch.cuda.OutOfMemoryError:
                print(f'[metrics] WARN {name!r} CUDA OOM; permanent fallback to CPU.')
                torch.cuda.empty_cache()
                if hasattr(metric, 'to'):
                    metric = metric.to('cpu')
                    self._metrics[name] = metric
                self._metric_devices[name] = 'cpu'
                t_cpu_again = t_cpu.to('cpu')
                with torch.inference_mode():
                    v = metric(t_cpu_again)
                out[name] = float(v.detach().float().cpu().item())
        return out

def compute_detection_metrics(det_raw: dict[str, Any], small_area_threshold: int=32 * 32) -> dict[str, float]:
    boxes = np.asarray(det_raw.get('boxes', []), dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(det_raw.get('scores', []), dtype=np.float32).reshape(-1)
    n = int(boxes.shape[0])
    if n == 0:
        return {'det_count': 0.0, 'mean_conf': float('nan'), 'high_conf_rate': float('nan'), 'small_obj_count': 0.0}
    w = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None)
    h = np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    areas = (w * h).astype(np.float32)
    return {'det_count': float(n), 'mean_conf': float(np.mean(scores)), 'high_conf_rate': float(np.mean(scores >= 0.5)), 'small_obj_count': float(int((areas < small_area_threshold).sum()))}

def compute_segmentation_metrics(mask: np.ndarray, num_classes: int=150, connectivity: int=8) -> dict[str, float]:
    if mask.ndim != 2:
        raise ValueError(f'Expected [H, W] mask, got {mask.shape}.')
    (h, w) = mask.shape
    total = float(h) * float(w)
    if total <= 0.0:
        return {'boundary_energy': 0.0, 'class_entropy': 0.0, 'n_components': 0.0}
    m_f32 = mask.astype(np.float32)
    gx = cv2.Sobel(m_f32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(m_f32, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.abs(gx) + np.abs(gy)
    boundary_energy = float(np.count_nonzero(grad)) / total
    (uniq, counts) = np.unique(mask, return_counts=True)
    probs = counts.astype(np.float64) / total
    probs = probs[probs > 0.0]
    ent = float(-np.sum(probs * np.log(probs)))
    max_ent = math.log(float(num_classes)) if num_classes > 1 else 1.0
    class_entropy = ent / max_ent if max_ent > 0.0 else 0.0
    total_components = 0
    for c in uniq.tolist():
        binary = (mask == c).astype(np.uint8)
        (num_labels, _) = cv2.connectedComponents(binary, connectivity=connectivity)
        total_components += max(0, int(num_labels) - 1)
    return {'boundary_energy': boundary_energy, 'class_entropy': float(class_entropy), 'n_components': float(total_components)}

def aggregate_per_method(rows: list[dict[str, Any]], metric_keys: tuple[str, ...]=ALL_METRIC_KEYS) -> dict[str, dict[str, dict[str, float]]]:
    by_method: dict[str, dict[str, dict[str, list[float]]]] = {}
    for r in rows:
        m = str(r['method'])
        c = str(r['clip'])
        bm = by_method.setdefault(m, {})
        bc = bm.setdefault(c, {})
        for k in metric_keys:
            v = r.get(k, float('nan'))
            if v is None:
                v = float('nan')
            bc.setdefault(k, []).append(float(v))
    agg: dict[str, dict[str, dict[str, float]]] = {}
    for (m, clips) in by_method.items():
        agg[m] = {}
        for k in metric_keys:
            clip_means: list[float] = []
            total_frames = 0
            for (_, d) in clips.items():
                vals = np.asarray(d.get(k, []), dtype=np.float64)
                total_frames += int(vals.size)
                if vals.size == 0:
                    continue
                if np.all(np.isnan(vals)):
                    continue
                clip_means.append(float(np.nanmean(vals)))
            if clip_means:
                arr = np.asarray(clip_means, dtype=np.float64)
                agg[m][k] = {'mean': float(np.mean(arr)), 'std': float(np.std(arr, ddof=0)), 'n_frames': int(total_frames), 'n_clips': int(arr.size)}
            else:
                agg[m][k] = {'mean': float('nan'), 'std': float('nan'), 'n_frames': int(total_frames), 'n_clips': 0}
    return agg
