from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Any
import numpy as np
import torch
YOLO_WEIGHTS: str = 'yolov8x.pt'
_LOCAL_WEIGHTS_DIR: Path = Path(__file__).resolve().parent / 'weights'
YOLO_CONF: float = 0.25
YOLO_IOU: float = 0.7
YOLO_IMGSZ: int = 1280
YOLO_AUGMENT: bool = False
YOLO_MAX_DET: int = 300
YOLO_VERBOSE: bool = False

def _resolve_weights(weights: str) -> str:
    p = Path(weights)
    if p.is_file():
        return str(p.resolve())
    local = _LOCAL_WEIGHTS_DIR / p.name
    if local.is_file():
        print(f'[YOLODetector] Using local weights: {local}')
        return str(local)
    return weights

def _seed_everything(seed: int=42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class YOLODetector:

    def __init__(self, weights: str=YOLO_WEIGHTS, device: str='cuda', seed: int=42) -> None:
        _seed_everything(seed)
        from ultralytics import YOLO
        self._YOLO_cls = YOLO
        self.device = device if torch.cuda.is_available() or device == 'cpu' else 'cpu'
        resolved = _resolve_weights(weights)
        self.model = YOLO(resolved)
        names = self.model.names
        if isinstance(names, dict):
            self.class_names: dict[int, str] = {int(k): v for (k, v) in names.items()}
        else:
            self.class_names = {i: n for (i, n) in enumerate(names)}

    @torch.inference_mode()
    def predict(self, image_bgr: np.ndarray) -> dict[str, Any]:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f'Expected [H,W,3] BGR uint8, got {image_bgr.shape}.')
        (h, w) = image_bgr.shape[:2]
        results = self.model.predict(image_bgr, conf=YOLO_CONF, iou=YOLO_IOU, imgsz=YOLO_IMGSZ, augment=YOLO_AUGMENT, max_det=YOLO_MAX_DET, verbose=YOLO_VERBOSE, device=self.device)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return {'boxes': np.zeros((0, 4), dtype=np.float32), 'scores': np.zeros((0,), dtype=np.float32), 'classes': np.zeros((0,), dtype=np.int32), 'class_names': self.class_names, 'image_shape': (h, w)}
        boxes = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = r.boxes.conf.detach().cpu().numpy().astype(np.float32)
        classes = r.boxes.cls.detach().cpu().numpy().astype(np.int32)
        return {'boxes': boxes, 'scores': scores, 'classes': classes, 'class_names': self.class_names, 'image_shape': (h, w)}

    @staticmethod
    def save_raw(result: dict[str, Any], out_path: str | Path) -> None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {'boxes': result['boxes'].tolist(), 'scores': result['scores'].tolist(), 'classes': result['classes'].tolist(), 'image_shape': list(result['image_shape'])}
        with out.open('w', encoding='utf-8') as f:
            json.dump(payload, f)

    @staticmethod
    def load_raw(in_path: str | Path, class_names: dict[int, str] | None=None) -> dict[str, Any]:
        with Path(in_path).open('r', encoding='utf-8') as f:
            payload = json.load(f)
        return {'boxes': np.asarray(payload['boxes'], dtype=np.float32).reshape(-1, 4), 'scores': np.asarray(payload['scores'], dtype=np.float32).reshape(-1), 'classes': np.asarray(payload['classes'], dtype=np.int32).reshape(-1), 'class_names': class_names or {}, 'image_shape': tuple(payload['image_shape'])}
