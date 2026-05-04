from __future__ import annotations
import random
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
SEGFORMER_MODEL_ID: str = 'nvidia/segformer-b5-finetuned-ade-640-640'
_LOCAL_WEIGHTS_DIR: Path = Path(__file__).resolve().parent / 'weights'
SEGFORMER_INPUT_SIZE: int = 640
SEGFORMER_NUM_CLASSES: int = 150

def _resolve_model_id(model_id: str) -> str:
    p = Path(model_id)
    if p.is_dir() and (p / 'config.json').exists():
        print(f'[SegFormerSegmenter] Using local model: {p}')
        return str(p.resolve())
    name = model_id.split('/')[-1] if '/' in model_id else model_id
    local = _LOCAL_WEIGHTS_DIR / name
    if local.is_dir() and (local / 'config.json').exists():
        print(f'[SegFormerSegmenter] Using local model: {local}')
        return str(local)
    return model_id

def _seed_everything(seed: int=42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SegFormerSegmenter:

    def __init__(self, model_id: str=SEGFORMER_MODEL_ID, device: str='cuda', fp16: bool=True, seed: int=42) -> None:
        _seed_everything(seed)
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        self.device = device if torch.cuda.is_available() or device == 'cpu' else 'cpu'
        self.fp16 = bool(fp16) and self.device.startswith('cuda')
        resolved = _resolve_model_id(model_id)
        self.processor = SegformerImageProcessor.from_pretrained(resolved)
        self.model = SegformerForSemanticSegmentation.from_pretrained(resolved)
        self.model.eval().to(self.device)
        raw_id2label = self.model.config.id2label
        self.id2label: dict[int, str] = {int(k): v for (k, v) in raw_id2label.items()}
        assert len(self.id2label) == SEGFORMER_NUM_CLASSES, f'Expected {SEGFORMER_NUM_CLASSES} classes, got {len(self.id2label)}.'

    @torch.inference_mode()
    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f'Expected [H,W,3] RGB uint8, got {image_rgb.shape}.')
        (h, w) = image_rgb.shape[:2]
        inputs = self.processor(images=[image_rgb], return_tensors='pt')
        pixel_values = inputs['pixel_values'].to(self.device)
        if self.fp16:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits.float()
        else:
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits
        upsampled = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        mask = upsampled.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
        assert mask.max() < SEGFORMER_NUM_CLASSES, f'Mask max {mask.max()} exceeds palette size.'
        return mask

    @staticmethod
    def save_raw(mask: np.ndarray, out_path: str | Path) -> None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        np.save(out, mask, allow_pickle=False)

    @staticmethod
    def load_raw(in_path: str | Path) -> np.ndarray:
        arr = np.load(in_path, allow_pickle=False)
        return arr.astype(np.uint8)
