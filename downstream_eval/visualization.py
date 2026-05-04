from __future__ import annotations
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from downstream_eval.palettes import get_display_font_path

def _auto_line_width(h: int, w: int) -> int:
    return max(2, int(min(h, w) / 500))

def _auto_font_size(h: int, w: int) -> int:
    return max(11, int(min(h, w) / 108))

def _load_font(size_px: int, bold: bool=True) -> ImageFont.ImageFont:
    try:
        path = get_display_font_path(size=size_px, bold=bold)
        return ImageFont.truetype(path, size=size_px)
    except Exception:
        return ImageFont.load_default()

def _rgba_from_palette(palette: np.ndarray, idx: int, alpha: int) -> tuple[int, int, int, int]:
    (r, g, b) = [int(v) for v in palette[int(idx) % len(palette)]]
    return (r, g, b, int(alpha))

def draw_detection_overlay(image_rgb: np.ndarray, result: dict[str, Any], class_names: dict[int, str], palette: np.ndarray, line_width: int | None=None, font_size: int | None=None, score_decimals: int=2) -> np.ndarray:
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3:
        raise ValueError(f'image_rgb must be [H,W,3] uint8, got {image_rgb.shape} {image_rgb.dtype}.')
    (h, w) = image_rgb.shape[:2]
    lw = int(line_width) if line_width is not None else _auto_line_width(h, w)
    fs = int(font_size) if font_size is not None else _auto_font_size(h, w)
    corner_radius = max(3, lw + 2)
    boxes = result.get('boxes', np.zeros((0, 4), dtype=np.float32))
    scores = result.get('scores', np.zeros((0,), dtype=np.float32))
    classes = result.get('classes', np.zeros((0,), dtype=np.int32))
    if len(boxes) == 0:
        return image_rgb.copy()
    order = np.argsort(scores)
    pil = Image.fromarray(image_rgb, mode='RGB').convert('RGBA')
    overlay = Image.new('RGBA', pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(fs, bold=True)
    pad_x = max(2, fs // 4)
    pad_y = max(2, fs // 6)
    for i in order:
        (x1, y1, x2, y2) = [float(v) for v in boxes[i]]
        x1 = max(0.0, min(float(w - 1), x1))
        y1 = max(0.0, min(float(h - 1), y1))
        x2 = max(0.0, min(float(w - 1), x2))
        y2 = max(0.0, min(float(h - 1), y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        cls_id = int(classes[i])
        score = float(scores[i])
        color_solid = _rgba_from_palette(palette, cls_id, alpha=255)
        draw.rounded_rectangle((x1, y1, x2, y2), radius=corner_radius, outline=color_solid, width=lw)
        label_text = f'{class_names.get(cls_id, str(cls_id))} {score:.{score_decimals}f}'
        try:
            tb = draw.textbbox((0, 0), label_text, font=font)
            (tw, th) = (tb[2] - tb[0], tb[3] - tb[1])
        except AttributeError:
            (tw, th) = draw.textsize(label_text, font=font)
        tag_h = th + 2 * pad_y
        tag_w = tw + 2 * pad_x
        tag_x1 = x1
        tag_y1 = y1 - tag_h
        if tag_y1 < 0:
            tag_y1 = y1
        tag_x2 = min(w - 1, tag_x1 + tag_w)
        tag_y2 = tag_y1 + tag_h
        tag_bg = _rgba_from_palette(palette, cls_id, alpha=int(0.85 * 255))
        draw.rounded_rectangle((tag_x1, tag_y1, tag_x2, tag_y2), radius=max(2, corner_radius // 2), fill=tag_bg, outline=None)
        draw.text((tag_x1 + pad_x, tag_y1 + pad_y), label_text, font=font, fill=(255, 255, 255, 255))
    composite = Image.alpha_composite(pil, overlay).convert('RGB')
    return np.asarray(composite, dtype=np.uint8)

def draw_segmentation_overlay(image_rgb: np.ndarray, mask: np.ndarray, palette: np.ndarray, alpha: float=0.5, draw_boundary: bool=True, boundary_thickness: int=2) -> np.ndarray:
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3:
        raise ValueError(f'image_rgb must be [H,W,3] uint8, got {image_rgb.shape}.')
    if mask.ndim != 2:
        raise ValueError(f'mask must be [H,W], got {mask.shape}.')
    if image_rgb.shape[:2] != mask.shape:
        raise ValueError(f'Shape mismatch: image {image_rgb.shape[:2]} vs mask {mask.shape}.')
    alpha = float(np.clip(alpha, 0.0, 1.0))
    color_mask = palette[np.clip(mask.astype(np.int32), 0, len(palette) - 1)]
    color_mask = color_mask.astype(np.uint8)
    blended = (1.0 - alpha) * image_rgb.astype(np.float32) + alpha * color_mask.astype(np.float32)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    if not draw_boundary:
        return blended
    mask_i16 = mask.astype(np.int16)
    gx = cv2.Sobel(mask_i16, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(mask_i16, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.abs(gx) + np.abs(gy) > 0
    if boundary_thickness > 1:
        kernel = np.ones((boundary_thickness, boundary_thickness), dtype=np.uint8)
        edge = cv2.dilate(edge.astype(np.uint8), kernel, iterations=1).astype(bool)
    blended[edge] = 0
    return blended

def save_subfigure(path: str | Path, overlay_rgb: np.ndarray, downsample: int=1, jpeg_quality: int=95) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if overlay_rgb.dtype != np.uint8 or overlay_rgb.ndim != 3:
        raise ValueError(f'overlay_rgb must be [H,W,3] uint8, got {overlay_rgb.shape}.')
    img = overlay_rgb
    if downsample and downsample > 1:
        (h, w) = img.shape[:2]
        new_hw = (w // downsample, h // downsample)
        img = cv2.resize(img, new_hw, interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if out.suffix.lower() in ('.jpg', '.jpeg'):
        cv2.imwrite(str(out), bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    else:
        cv2.imwrite(str(out), bgr)
