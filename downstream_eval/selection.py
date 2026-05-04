from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from downstream_eval.detection import YOLODetector

def _load_det_raw(raw_root: Path, method: str, clip: str, frame_stem: str) -> dict[str, Any] | None:
    p = raw_root / method / clip / f'{frame_stem}.json'
    if not p.exists():
        return None
    return YOLODetector.load_raw(p)

def select_top_scenes(raw_det_root: Path, reference_method: str='hazy', target_method: str='ours', top_k: int=5, min_ref_boxes: int=0) -> list[dict[str, Any]]:
    raw_det_root = Path(raw_det_root)
    ref_root = raw_det_root / reference_method
    tgt_root = raw_det_root / target_method
    if not ref_root.exists():
        raise FileNotFoundError(f'Reference raw dir not found: {ref_root}')
    if not tgt_root.exists():
        raise FileNotFoundError(f'Target raw dir not found: {tgt_root}')
    candidates: list[dict[str, Any]] = []
    for clip_dir in sorted((p for p in tgt_root.iterdir() if p.is_dir())):
        clip = clip_dir.name
        for frame_json in sorted(clip_dir.glob('*.json')):
            frame_stem = frame_json.stem
            tgt = _load_det_raw(raw_det_root, target_method, clip, frame_stem)
            ref = _load_det_raw(raw_det_root, reference_method, clip, frame_stem)
            if tgt is None or ref is None:
                continue
            n_tgt = int(len(tgt['boxes']))
            n_ref = int(len(ref['boxes']))
            if n_ref < min_ref_boxes:
                continue
            delta = n_tgt - n_ref
            delta_score = float(tgt['scores'].sum()) - float(ref['scores'].sum())
            candidates.append({'clip': clip, 'frame': frame_stem, 'delta': delta, 'n_ref': n_ref, 'n_target': n_tgt, 'delta_score': round(delta_score, 4)})
    if not candidates:
        raise RuntimeError(f'No overlapping (clip, frame) samples between {reference_method} and {target_method} under {raw_det_root}.')
    candidates.sort(key=lambda r: (-r['delta'], -r['delta_score'], r['clip'], r['frame']))
    return candidates[:int(top_k)]

def write_scenes_json(scenes: list[dict[str, Any]], out_path: str | Path, metadata: dict[str, Any] | None=None) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {'metadata': metadata or {}, 'scenes': scenes}
    with out.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def read_scenes_json(in_path: str | Path) -> list[dict[str, Any]]:
    with Path(in_path).open('r', encoding='utf-8') as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    return list(payload.get('scenes', []))

def parse_scene_spec(spec: str) -> list[dict[str, Any]]:
    spec = spec.strip()
    if not spec:
        return []
    p = Path(spec)
    if p.exists() and p.is_file():
        return read_scenes_json(p)
    scenes: list[dict[str, Any]] = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(f"Scene spec '{item}' missing ':' separator. Expected format: clip_name:frame_stem (e.g. C002_1:00042).")
        (clip, frame) = item.split(':', 1)
        scenes.append({'clip': clip.strip(), 'frame': frame.strip()})
    return scenes
