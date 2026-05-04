from __future__ import annotations
import csv
import math
from pathlib import Path
from typing import Any
from downstream_eval.metrics import ALL_METRIC_KEYS, DETECTION_METRIC_KEYS, IMAGE_METRIC_KEYS, METRIC_DIRECTION, METRIC_PRETTY, SEGMENTATION_METRIC_KEYS
from downstream_eval.palettes import pretty_name
METRIC_PRECISION: dict[str, int] = {'niqe': 2, 'brisque': 2, 'musiq': 2, 'det_count': 1, 'mean_conf': 3, 'high_conf_rate': 3, 'small_obj_count': 1, 'boundary_energy': 4, 'class_entropy': 3, 'n_components': 1}
ORACLE_METHODS: frozenset[str] = frozenset({'gt'})
DEFAULT_OURS_ROW_COLOR: str = 'ourgreen'

def _fmt_value(value: float, precision: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '-'
    return f'{value:.{precision}f}'

def _direction_arrow(key: str, markup: str='unicode') -> str:
    d = METRIC_DIRECTION.get(key, 'up')
    if markup == 'latex':
        return '$\\downarrow$' if d == 'down' else '$\\uparrow$'
    return '↓' if d == 'down' else '↑'

def _rank_column(agg: dict[str, dict[str, dict[str, float]]], methods: list[str], metric_key: str) -> tuple[float | None, float | None]:
    direction = METRIC_DIRECTION.get(metric_key, 'up')
    values: list[float] = []
    for m in methods:
        if m in ORACLE_METHODS:
            continue
        entry = agg.get(m, {}).get(metric_key)
        if entry is None:
            continue
        v = entry.get('mean', float('nan'))
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        values.append(float(v))
    if not values:
        return (None, None)
    uniq_sorted = sorted(set(values), reverse=direction == 'up')
    best = uniq_sorted[0]
    second = uniq_sorted[1] if len(uniq_sorted) > 1 else None
    return (best, second)

def _is_close(a: float, b: float | None, *, abs_tol: float=1e-06) -> bool:
    if b is None:
        return False
    if math.isnan(a) or math.isnan(b):
        return False
    return math.isclose(a, b, rel_tol=1e-06, abs_tol=abs_tol)

def write_per_frame_csv(rows: list[dict[str, Any]], out_path: str | Path, metric_keys: tuple[str, ...]=ALL_METRIC_KEYS) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ['method', 'clip', 'frame', *metric_keys]
    with out.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            line = [r.get('method', ''), r.get('clip', ''), r.get('frame', '')]
            for k in metric_keys:
                v = r.get(k, '')
                if isinstance(v, float) and math.isnan(v):
                    line.append('nan')
                else:
                    line.append(v)
            writer.writerow(line)
    return out

def write_per_method_summary_csv(agg: dict[str, dict[str, dict[str, float]]], out_path: str | Path, methods: list[str] | None=None, metric_keys: tuple[str, ...]=ALL_METRIC_KEYS) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    method_list = methods if methods is not None else list(agg.keys())
    header = ['method', 'metric', 'mean', 'std', 'n_frames', 'n_clips']
    with out.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for m in method_list:
            for k in metric_keys:
                entry = agg.get(m, {}).get(k, {})
                mean = entry.get('mean', float('nan'))
                std = entry.get('std', float('nan'))
                n_frames = int(entry.get('n_frames', 0))
                n_clips = int(entry.get('n_clips', 0))
                writer.writerow([m, k, 'nan' if isinstance(mean, float) and math.isnan(mean) else f'{mean:.6g}', 'nan' if isinstance(std, float) and math.isnan(std) else f'{std:.6g}', n_frames, n_clips])
    return out

def write_markdown_table(agg: dict[str, dict[str, dict[str, float]]], out_path: str | Path, methods: list[str], metric_keys: tuple[str, ...]=ALL_METRIC_KEYS, dataset_name: str | None=None) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rank_cache: dict[str, tuple[float | None, float | None]] = {k: _rank_column(agg, methods, k) for k in metric_keys}
    head_cells = ['Method'] + [f'{METRIC_PRETTY.get(k, k)} {_direction_arrow(k)}' for k in metric_keys]
    sep_cells = ['---'] * len(head_cells)
    body_rows: list[list[str]] = []
    for m in methods:
        label = pretty_name(m)
        if m == 'ours':
            label = f'**{label}**'
        elif m in ORACLE_METHODS:
            label = f'*{label}*'
        row = [label]
        for k in metric_keys:
            entry = agg.get(m, {}).get(k, {})
            value = entry.get('mean', float('nan'))
            precision = METRIC_PRECISION.get(k, 3)
            cell = _fmt_value(value, precision)
            if cell == '-' or m in ORACLE_METHODS:
                row.append(cell)
                continue
            (best, second) = rank_cache[k]
            if _is_close(float(value), best):
                cell = f'**{cell}**'
            elif _is_close(float(value), second):
                cell = f'*{cell}*'
            row.append(cell)
        body_rows.append(row)
    title = 'Downstream NR metrics'
    if dataset_name:
        title += f' — {dataset_name}'
    lines: list[str] = [f'## {title}', '', 'Best values in **bold**, second best in *italic*. Oracle rows (*GT*) are excluded from ranking and shown as an upper reference for the frozen detectors.', '', '| ' + ' | '.join(head_cells) + ' |', '| ' + ' | '.join(sep_cells) + ' |']
    for row in body_rows:
        lines.append('| ' + ' | '.join(row) + ' |')
    lines.append('')
    out.write_text('\n'.join(lines), encoding='utf-8')
    return out

def write_latex_table(agg: dict[str, dict[str, dict[str, float]]], out_path: str | Path, methods: list[str], metric_keys: tuple[str, ...]=ALL_METRIC_KEYS, dataset_name: str | None=None, ours_row_color: str=DEFAULT_OURS_ROW_COLOR, caption: str | None=None, label: str='tab:downstream_nr') -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rank_cache: dict[str, tuple[float | None, float | None]] = {k: _rank_column(agg, methods, k) for k in metric_keys}
    img_keys = [k for k in metric_keys if k in IMAGE_METRIC_KEYS]
    det_keys = [k for k in metric_keys if k in DETECTION_METRIC_KEYS]
    seg_keys = [k for k in metric_keys if k in SEGMENTATION_METRIC_KEYS]
    ordered_keys = img_keys + det_keys + seg_keys
    col_spec_parts = ['l']
    group_headers: list[tuple[str, int, bool]] = []
    if img_keys:
        col_spec_parts.append('c' * len(img_keys))
        group_headers.append(('Image quality (NR)', len(img_keys), True))
    if det_keys:
        col_spec_parts.append('c' * len(det_keys))
        group_headers.append(('Detection (NR)', len(det_keys), True))
    if seg_keys:
        col_spec_parts.append('c' * len(seg_keys))
        group_headers.append(('Segmentation (NR)', len(seg_keys), False))
    col_spec_parts_with_rules: list[str] = [col_spec_parts[0]]
    for (i, part) in enumerate(col_spec_parts[1:]):
        col_spec_parts_with_rules.append(part)
        is_last = i == len(col_spec_parts) - 2
        if not is_last:
            col_spec_parts_with_rules.append('|')
    col_spec = ' '.join(col_spec_parts_with_rules)
    caption = caption or 'No-reference quantitative metrics for the downstream task evaluation. Lower is better for NIQE and BRISQUE; higher is better for all other columns. Best results in \\textbf{bold}, second best \\underline{underlined}; ranking excludes the GT (oracle) row, which bounds the frozen detectors.'
    if dataset_name:
        caption = f'[{dataset_name}] ' + caption
    lines: list[str] = []
    lines.append('% Auto-generated by downstream_eval/metrics_report.py')
    lines.append('% Requires: \\usepackage{booktabs} and ideally')
    lines.append(f'%           \\usepackage[table]{{xcolor}} + \\definecolor{{{ours_row_color}}}{{...}}')
    lines.append('\\begin{table}[t]')
    lines.append('  \\centering')
    lines.append('  \\small')
    lines.append(f'  \\caption{{{caption}}}')
    lines.append(f'  \\label{{{label}}}')
    lines.append('  \\resizebox{\\linewidth}{!}{%')
    lines.append(f'  \\begin{{tabular}}{{{col_spec}}}')
    lines.append('    \\toprule')
    group_row_cells: list[str] = [' ']
    for (title, span, draw_rule) in group_headers:
        align = f"c{('|' if draw_rule else '')}"
        group_row_cells.append(f'\\multicolumn{{{span}}}{{{align}}}{{{title}}}')
    lines.append('    ' + ' & '.join(group_row_cells) + ' \\\\')
    per_metric_cells = ['Method']
    for k in ordered_keys:
        per_metric_cells.append(f"{METRIC_PRETTY.get(k, k)}\\,{_direction_arrow(k, markup='latex')}")
    lines.append('    ' + ' & '.join(per_metric_cells) + ' \\\\')
    lines.append('    \\midrule')
    for m in methods:
        label_cells: list[str] = []
        label_text = pretty_name(m).replace('&', '\\&')
        if m == 'ours':
            label_cells.append(f'\\textbf{{{label_text}}}')
        elif m in ORACLE_METHODS:
            label_cells.append(f'\\textit{{{label_text}}}')
        else:
            label_cells.append(label_text)
        for k in ordered_keys:
            entry = agg.get(m, {}).get(k, {})
            value = entry.get('mean', float('nan'))
            precision = METRIC_PRECISION.get(k, 3)
            cell = _fmt_value(value, precision)
            if cell == '-' or m in ORACLE_METHODS:
                if m in ORACLE_METHODS and cell != '-':
                    cell = f'\\textit{{{cell}}}'
                label_cells.append(cell)
                continue
            (best, second) = rank_cache[k]
            if _is_close(float(value), best):
                cell = f'\\textbf{{{cell}}}'
            elif _is_close(float(value), second):
                cell = f'\\underline{{{cell}}}'
            label_cells.append(cell)
        row_line = '    ' + ' & '.join(label_cells) + ' \\\\'
        if m == 'ours' and ours_row_color:
            row_line = f'    \\rowcolor{{{ours_row_color}}}\n' + row_line
        lines.append(row_line)
    lines.append('    \\bottomrule')
    lines.append('  \\end{tabular}}')
    lines.append('\\end{table}')
    lines.append('')
    out.write_text('\n'.join(lines), encoding='utf-8')
    return out

def write_all(rows: list[dict[str, Any]], agg: dict[str, dict[str, dict[str, float]]], out_dir: str | Path, methods: list[str], metric_keys: tuple[str, ...]=ALL_METRIC_KEYS, dataset_name: str | None=None, export_latex: bool=True) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    written['per_frame_csv'] = write_per_frame_csv(rows, out_dir / 'per_frame.csv', metric_keys=metric_keys)
    written['per_method_summary_csv'] = write_per_method_summary_csv(agg, out_dir / 'per_method_summary.csv', methods=methods, metric_keys=metric_keys)
    written['markdown_table'] = write_markdown_table(agg, out_dir / 'table.md', methods=methods, metric_keys=metric_keys, dataset_name=dataset_name)
    if export_latex:
        written['latex_table'] = write_latex_table(agg, out_dir / 'table.tex', methods=methods, metric_keys=metric_keys, dataset_name=dataset_name)
    return written
