from __future__ import annotations
import colorsys
import numpy as np
from matplotlib import colormaps
from matplotlib.font_manager import FontProperties, findfont
ADE20K_PALETTE: np.ndarray = np.array([[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3], [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7], [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255], [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255], [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255], [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20], [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255], [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0], [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0], [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112], [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163], [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0], [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0], [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255], [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204], [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255], [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255], [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194], [102, 255, 0], [92, 0, 255]], dtype=np.uint8)
assert ADE20K_PALETTE.shape == (150, 3), 'ADE20K palette must be 150x3 uint8.'

def _build_coco_palette() -> np.ndarray:
    base = list(colormaps['tab20'].colors) + list(colormaps['tab20b'].colors) + list(colormaps['tab20c'].colors)
    rgb = np.asarray(base, dtype=np.float32)[:, :3]
    boosted: list[list[float]] = []
    for color in colormaps['tab20'].colors:
        (r, g, b) = (float(color[0]), float(color[1]), float(color[2]))
        (h, s, v) = colorsys.rgb_to_hsv(r, g, b)
        v = min(1.0, v + 0.15)
        s = min(1.0, s * 1.1)
        (br, bg, bb) = colorsys.hsv_to_rgb(h, s, v)
        boosted.append([br, bg, bb])
    boosted_arr = np.asarray(boosted, dtype=np.float32)
    palette = np.concatenate([rgb, boosted_arr], axis=0)
    palette = np.clip(palette * 255.0, 0, 255).astype(np.uint8)
    assert palette.shape == (80, 3)
    return palette
COCO_DETECTION_PALETTE: np.ndarray = _build_coco_palette()
METHOD_PRETTY_NAMES: dict[str, str] = {'hazy': 'Hazy', 'gt': 'GT (oracle)', 'ours': 'Ours'}

def pretty_name(method: str) -> str:
    if method in METHOD_PRETTY_NAMES:
        return METHOD_PRETTY_NAMES[method]
    return method.replace('_', ' ').title()
_DISPLAY_FONT_STACK: list[str] = ['Helvetica', 'Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']

def get_display_font(size: int=12, weight: str='bold') -> FontProperties:
    fp = FontProperties(family=_DISPLAY_FONT_STACK, size=size, weight=weight)
    try:
        findfont(fp, fallback_to_default=True)
    except Exception:
        pass
    return fp

def get_display_font_path(size: int=12, bold: bool=True) -> str:
    weight = 'bold' if bold else 'normal'
    fp = FontProperties(family=_DISPLAY_FONT_STACK, size=size, weight=weight)
    return findfont(fp, fallback_to_default=True)
