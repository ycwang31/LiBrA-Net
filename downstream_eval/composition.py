from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patheffects import Stroke, Normal
from downstream_eval.io_utils import imread_rgb
from downstream_eval.palettes import pretty_name
from downstream_eval.style import apply_publication_style
SECTION_TITLE_FONTSIZE: float = 11.0
COL_LABEL_FONTSIZE: float = 10.0
ROW_LABEL_FONTSIZE: float = 10.0
TASK_TITLES: dict[str, str] = {'detection': '(a) Object Detection — YOLOv8-X (COCO pretrained)', 'segmentation': '(b) Semantic Segmentation — SegFormer-B5 (ADE20K pretrained)'}

def _styled_column_label(ax: plt.Axes, method: str, y: float=1.03) -> None:
    label = pretty_name(method)
    if method == 'ours':
        t = ax.set_title(label, fontsize=COL_LABEL_FONTSIZE, fontweight='bold', y=y, color='#111111')
        t.set_path_effects([Stroke(linewidth=0.4, foreground='#111111'), Normal()])
    elif method == 'gt':
        ax.set_title(label, fontsize=COL_LABEL_FONTSIZE, fontstyle='italic', y=y, color='#444444')
    else:
        ax.set_title(label, fontsize=COL_LABEL_FONTSIZE, y=y, color='#111111')

def _underline_ours(ax: plt.Axes, method: str) -> None:
    if method != 'ours':
        return
    line = Line2D([0.25, 0.75], [1.005, 1.005], color='#111111', linewidth=0.9, clip_on=False)
    line.set_transform(ax.transAxes)
    ax.add_artist(line)

def _tasks_for(task: str) -> list[str]:
    if task == 'both':
        return ['detection', 'segmentation']
    if task in ('detection', 'segmentation'):
        return [task]
    raise ValueError(f"task must be 'detection', 'segmentation', or 'both'; got {task!r}.")

def _read_overlay(viz_root: Path, task: str, method: str, clip: str, frame_stem: str) -> np.ndarray:
    p = viz_root / task / method / clip / f'{frame_stem}.png'
    if not p.exists():
        for ext in ('.jpg', '.jpeg'):
            alt = p.with_suffix(ext)
            if alt.exists():
                p = alt
                break
        else:
            raise FileNotFoundError(f'Overlay not found: {p}. Run `run_downstream_eval.py infer` first to populate viz/.')
    return imread_rgb(p)

def compose_figure(dataset_name: str, scene: dict[str, Any], methods: list[str], task: str, viz_root: Path, out_path_stem: Path, dpi: int=300, col_inch: float=2.0, row_aspect: float=9.0 / 16.0, exports: Iterable[str]=('pdf', 'png'), use_latex: bool=False) -> list[Path]:
    apply_publication_style(use_latex=use_latex)
    tasks = _tasks_for(task)
    n_rows = len(tasks)
    n_cols = len(methods)
    if n_cols < 1:
        raise ValueError('Need at least one method column.')
    if n_cols > 8:
        print(f'[compose_figure] Warning: {n_cols} columns may overflow a single-page figure; consider splitting or reducing method list.')
    section_reserve_in = 0.45
    fig_width = col_inch * n_cols + 0.6
    per_row_image_height = col_inch * row_aspect
    fig_height = (per_row_image_height + section_reserve_in) * n_rows + 0.15
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')
    clip = scene['clip']
    frame_stem = scene['frame']
    strip_height = 1.0 / n_rows
    title_band_frac = section_reserve_in / (per_row_image_height + section_reserve_in)
    for (row_idx, t) in enumerate(tasks):
        strip_top = 1.0 - row_idx * strip_height
        strip_bottom = strip_top - strip_height
        images_top = strip_top - strip_height * title_band_frac
        images_top = max(images_top - 0.01, strip_bottom + 0.02)
        gs = GridSpec(nrows=1, ncols=n_cols, left=0.04, right=0.995, top=images_top, bottom=strip_bottom + 0.02, wspace=0.02, figure=fig)
        section_title_y = strip_top - 0.25 * strip_height * title_band_frac
        fig.text(0.5, section_title_y, TASK_TITLES[t], ha='center', va='center', fontsize=SECTION_TITLE_FONTSIZE, fontweight='bold', color='#111111')
        for (col_idx, method) in enumerate(methods):
            ax = fig.add_subplot(gs[0, col_idx])
            try:
                img = _read_overlay(viz_root, t, method, clip, frame_stem)
            except FileNotFoundError as e:
                ax.text(0.5, 0.5, f'missing\n{method}', ha='center', va='center', fontsize=7, color='#aa0000', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                print(f'[compose_figure] {e}')
                continue
            ax.imshow(img, interpolation='antialiased')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            _styled_column_label(ax, method, y=1.03)
            _underline_ours(ax, method)
    out_path_stem = Path(out_path_stem)
    out_path_stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in exports:
        out_path = out_path_stem.with_suffix(f'.{ext}')
        fig.savefig(out_path, dpi=dpi)
        written.append(out_path)
    plt.close(fig)
    return written
