from __future__ import annotations
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import yaml
IMAGE_EXTS: tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG')

def load_config(config_path: str | Path) -> dict[str, Any]:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f'Config not found: {p}')
    with p.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f'Config root must be a mapping, got {type(cfg).__name__}.')
    cfg.setdefault('_meta', {})
    cfg['_meta']['dataset_name'] = p.stem
    cfg['_meta']['config_path'] = str(p.resolve())
    return cfg

def _resolve_split_dir(root: Path, split: str) -> Path:
    for name in (split, split.capitalize(), split.lower(), split.upper()):
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Split "{split}" not found under {root}. Configure data.test_split explicitly.')

def enumerate_frames(root: Path) -> list[tuple[str, str, str]]:
    if not root.exists():
        return []
    results: list[tuple[str, str, str]] = []
    for clip_dir in sorted((p for p in root.iterdir() if p.is_dir())):
        for frame in sorted(clip_dir.iterdir()):
            if frame.suffix in IMAGE_EXTS:
                results.append((clip_dir.name, frame.stem, frame.suffix))
    return results

def frame_path(root: Path, clip: str, frame_stem: str, ext_candidates: tuple[str, ...]=IMAGE_EXTS) -> Path:
    for ext in ext_candidates:
        p = root / clip / f'{frame_stem}{ext}'
        if p.exists():
            return p
    raise FileNotFoundError(f'Frame {frame_stem} not found under {root / clip} with any of {ext_candidates}.')
DATASET_ROLE_METHODS: frozenset[str] = frozenset({'hazy', 'gt'})
OURS_METHOD: str = 'ours'
KNOWN_METHODS: frozenset[str] = DATASET_ROLE_METHODS | {OURS_METHOD}

def resolve_method_input_dir(method: str, config: dict[str, Any], dehazed_root: Path) -> Path:
    if method not in KNOWN_METHODS:
        raise ValueError(f"Unknown method '{method}'. Known methods: {sorted(KNOWN_METHODS)}.")
    data_cfg = config.get('data', {})
    dataset_root = Path(data_cfg.get('dataset_root', ''))
    test_split = data_cfg.get('test_split', 'test')
    if method in DATASET_ROLE_METHODS:
        if not dataset_root.parts:
            raise ValueError('config.data.dataset_root must be set for hazy/gt.')
        split_dir = _resolve_split_dir(dataset_root, test_split)
        path = split_dir / method
    else:
        path = dehazed_root / method
    if not path.exists():
        if method in DATASET_ROLE_METHODS:
            hint = f"Dataset role '{method}' input missing. Check that {dataset_root}/{test_split}/{method} exists."
        else:
            hint = f"Method '{method}' dehazed output missing. Run: python test.py --config {config['_meta']['config_path']} --model {method} --save_images"
        raise FileNotFoundError(f"Method '{method}' input directory {path} not found.\n{hint}")
    return path

def validate_methods(methods: list[str], config: dict[str, Any], dehazed_root: Path) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for method in methods:
        path = resolve_method_input_dir(method, config, dehazed_root)
        clips = [p for p in path.iterdir() if p.is_dir()]
        if not clips:
            raise FileNotFoundError(f"Method '{method}' directory {path} exists but has no clip subdirs.")
        resolved[method] = path
    return resolved

def validate_resolution_consistency(method_dirs: dict[str, Path], sample_clip: str | None=None, sample_frame: str | None=None) -> tuple[int, int]:
    if not method_dirs:
        raise ValueError('method_dirs is empty, nothing to validate.')
    first_method = next(iter(method_dirs))
    first_root = method_dirs[first_method]
    clips = sorted((p for p in first_root.iterdir() if p.is_dir()))
    if not clips:
        raise FileNotFoundError(f'No clips under {first_root}.')
    if sample_clip is None:
        sample_clip = clips[0].name
    if sample_frame is None:
        frames = sorted((p for p in (first_root / sample_clip).iterdir() if p.suffix in IMAGE_EXTS))
        if not frames:
            raise FileNotFoundError(f'No frames in {first_root / sample_clip}.')
        sample_frame = frames[0].stem
    shapes: dict[str, tuple[int, int]] = {}
    for (method, root) in method_dirs.items():
        try:
            fp = frame_path(root, sample_clip, sample_frame)
        except FileNotFoundError:
            continue
        img = cv2.imread(str(fp))
        if img is None:
            raise RuntimeError(f'cv2.imread failed for {fp}.')
        shapes[method] = img.shape[:2]
    if not shapes:
        raise RuntimeError(f'No method had a readable image for sample {sample_clip}/{sample_frame}.')
    ref = next(iter(shapes.values()))
    mismatches = {m: s for (m, s) in shapes.items() if s != ref}
    if mismatches:
        raise RuntimeError(f'Resolution inconsistency detected at {sample_clip}/{sample_frame}: reference {ref}, mismatches {mismatches}. All outputs must have identical resolution.')
    return ref

def imread_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'cv2.imread failed for {path}.')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def imwrite_rgb(path: str | Path, image_rgb: np.ndarray, jpeg_quality: int=95) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if out.suffix.lower() in ('.jpg', '.jpeg'):
        cv2.imwrite(str(out), bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    else:
        cv2.imwrite(str(out), bgr)
