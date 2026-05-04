from __future__ import annotations
import argparse
import csv
import gc
import importlib
import statistics
import sys
import time
from collections import OrderedDict
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
import torch
import torch.nn as nn
_T_DEFAULT = 5
_FPS_RESOLUTIONS: OrderedDict[str, tuple[int, int]] = OrderedDict([('720p', (1280, 720)), ('1080p', (1920, 1080)), ('2K', (2560, 1440)), ('2.7K', (2704, 1520)), ('4K', (3840, 2160)), ('6K', (5760, 3240)), ('8K', (7680, 4320))])
_PEAK_MEM_RES_NAME = '4K'
_PEAK_MEM_RES: tuple[int, int] = (3840, 2160)
_MODEL_REGISTRY: OrderedDict[str, tuple[str, str, str, dict]] = OrderedDict([('Ours', ('models', 'BilateralGridDehazeNet', 'video', {}))])
_CUDA = torch.cuda.is_available()

def _sync() -> None:
    if _CUDA:
        torch.cuda.synchronize()

def _reset_peak() -> None:
    if _CUDA:
        torch.cuda.reset_peak_memory_stats()

def _peak_gb() -> float:
    if _CUDA:
        return torch.cuda.max_memory_allocated() / 1024 ** 3
    return 0.0

def _bench(fn, num_warmup: int=5, num_bench: int=20) -> dict[str, float]:
    for _ in range(num_warmup):
        fn()
        _sync()
    if _CUDA:
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)
        ts: list[float] = []
        for _ in range(num_bench):
            torch.cuda.synchronize()
            ev_s.record()
            fn()
            ev_e.record()
            torch.cuda.synchronize()
            ts.append(ev_s.elapsed_time(ev_e))
    else:
        ts = []
        for _ in range(num_bench):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000.0)
    return {'mean': statistics.mean(ts), 'std': statistics.stdev(ts) if len(ts) > 1 else 0.0, 'min': min(ts), 'max': max(ts)}

def _build_model(module_path: str, class_name: str, device: torch.device, ctor_kwargs: dict | None=None) -> nn.Module:
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    model = cls(**ctor_kwargs or {})
    model.to(device).eval()
    return model

def _count_params_m(model: nn.Module) -> float:
    return sum((p.numel() for p in model.parameters())) / 1000000.0

def _cleanup(model: nn.Module | None, *tensors) -> None:
    for t in tensors:
        del t
    if model is not None:
        del model
    gc.collect()
    if _CUDA:
        torch.cuda.empty_cache()
_NATIVE_ENCODE_SIZE = 999999

def _measure_peak_mem(model: nn.Module, T: int, res_name: str, W: int, H: int, device: torch.device) -> str:
    x: torch.Tensor | None = None
    try:
        x = torch.rand(1, T, 3, H, W, device=device)
        _reset_peak()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=_CUDA):
            model.inference(x, H, W, encode_size=_NATIVE_ENCODE_SIZE)
        _sync()
        gb = _peak_gb()
        return f'{gb:.2f}'
    except RuntimeError as e:
        if 'out of memory' in str(e).lower() or 'CUDA out of memory' in str(e):
            return 'OOM'
        return f'ERR:{type(e).__name__}'
    finally:
        if x is not None:
            del x
        gc.collect()
        if _CUDA:
            torch.cuda.empty_cache()

def _measure_fps(model: nn.Module, T: int, res_name: str, W: int, H: int, device: torch.device, num_warmup: int, num_bench: int) -> str:
    try:
        x = torch.rand(1, T, 3, H, W, device=device)
    except RuntimeError:
        return 'OOM'

    def _fn(_x=x, _h=H, _w=W) -> None:
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=_CUDA):
            model.inference(_x, _h, _w, encode_size=_NATIVE_ENCODE_SIZE)
        _sync()
    try:
        stats = _bench(_fn, num_warmup=num_warmup, num_bench=num_bench)
        fps = 1000.0 / stats['mean'] if stats['mean'] > 0 else float('inf')
        return f'{fps:.2f}'
    except RuntimeError as e:
        if 'out of memory' in str(e).lower() or 'CUDA out of memory' in str(e):
            return 'OOM'
        return f'ERR:{type(e).__name__}'
    finally:
        del x
        gc.collect()
        if _CUDA:
            torch.cuda.empty_cache()

def run_benchmark(model_names: list[str], num_frames: int, num_warmup: int, num_bench: int, skip_peakmem: bool, skip_fps: bool, output_path: Path) -> list[dict]:
    device = torch.device('cuda' if _CUDA else 'cpu')
    if _CUDA:
        props = torch.cuda.get_device_properties(0)
        print(f'Device:  {torch.cuda.get_device_name(0)}')
        print(f'VRAM:    {props.total_memory / 1024 ** 3:.1f} GB')
    else:
        print('Device:  CPU (no CUDA — FPS will be slow)')
    print(f'T (frames): {num_frames}  warmup={num_warmup}  bench={num_bench}')
    print(f'Models:  {len(model_names)}')
    print()
    fps_cols = list(_FPS_RESOLUTIONS.keys())
    col_headers = ['Method', 'Type', 'Params(M)']
    if not skip_peakmem:
        col_headers += [f'PeakMem@{_PEAK_MEM_RES_NAME}(GB)']
    if not skip_fps:
        col_headers += [f'FPS@{r}' for r in fps_cols]
    rows: list[dict] = []
    total = len(model_names)
    for (idx, name) in enumerate(model_names, 1):
        if name not in _MODEL_REGISTRY:
            print(f'[{idx}/{total}] SKIP  {name}  (not found in registry)')
            continue
        (module_path, class_name, type_label, ctor_kwargs) = _MODEL_REGISTRY[name]
        print(f'[{idx}/{total}] {name}  ({module_path}.{class_name})')
        if ctor_kwargs:
            print(f'         ctor_kwargs: {ctor_kwargs}')
        model: nn.Module | None = None
        try:
            model = _build_model(module_path, class_name, device, ctor_kwargs)
            params_m = _count_params_m(model)
            print(f'         Params: {params_m:.2f}M')
        except Exception as e:
            print(f'         LOAD ERROR: {e}')
            _cleanup(model)
            row = {'Method': name, 'Type': type_label, 'Params(M)': 'ERR'}
            if not skip_peakmem:
                row[f'PeakMem@{_PEAK_MEM_RES_NAME}(GB)'] = 'ERR'
            if not skip_fps:
                for r in fps_cols:
                    row[f'FPS@{r}'] = 'ERR'
            rows.append(row)
            continue
        row: dict = {'Method': name, 'Type': type_label, 'Params(M)': f'{params_m:.2f}'}
        if not skip_peakmem:
            (W4K, H4K) = _PEAK_MEM_RES
            print(f'         Peak Mem @{_PEAK_MEM_RES_NAME} ({W4K}×{H4K}) ...', end=' ', flush=True)
            result = _measure_peak_mem(model, num_frames, _PEAK_MEM_RES_NAME, W4K, H4K, device)
            print(result)
            row[f'PeakMem@{_PEAK_MEM_RES_NAME}(GB)'] = result
        if not skip_fps:
            for (res_name, (W, H)) in _FPS_RESOLUTIONS.items():
                print(f'         FPS @{res_name} ({W}×{H}) ...', end=' ', flush=True)
                result = _measure_fps(model, num_frames, res_name, W, H, device, num_warmup, num_bench)
                print(result)
                row[f'FPS@{res_name}'] = result
        rows.append(row)
        _cleanup(model)
        print()
    return (rows, col_headers)

def _fmt_val(v: str, width: int) -> str:
    return v.ljust(width)

def print_markdown_table(rows: list[dict], col_headers: list[str]) -> None:
    if not rows:
        print('(no results)')
        return
    widths = {h: len(h) for h in col_headers}
    for row in rows:
        for h in col_headers:
            widths[h] = max(widths[h], len(str(row.get(h, ''))))
    header = '| ' + ' | '.join((h.ljust(widths[h]) for h in col_headers)) + ' |'
    sep = '|-' + '-|-'.join(('-' * widths[h] for h in col_headers)) + '-|'
    print(header)
    print(sep)
    for row in rows:
        line = '| ' + ' | '.join((str(row.get(h, '')).ljust(widths[h]) for h in col_headers)) + ' |'
        print(line)

def save_csv(rows: list[dict], col_headers: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=col_headers, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nCSV saved → {path}')

def parse_args() -> argparse.Namespace:
    all_names = list(_MODEL_REGISTRY.keys())
    p = argparse.ArgumentParser(description='Benchmark Peak Mem @4K + FPS @720p/1080p/2K/2.7K for the released model.\nConditions: B=1, T=5, FP16 autocast, CUDA Event timing, warmup=5, bench=20.', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='Examples:\n  python scripts/benchmark_efficiency.py\n  python scripts/benchmark_efficiency.py --models Ours\n  python scripts/benchmark_efficiency.py --skip_peakmem\n  python scripts/benchmark_efficiency.py --num_bench 50 --output results/bench.csv\n')
    p.add_argument('--models', nargs='+', default=None, metavar='NAME', help=f'Models to test (default: all {len(all_names)}). Choices: {all_names}')
    p.add_argument('--num_warmup', type=int, default=5, help='Warmup iterations before timing (default: 5, same as testmodel.py)')
    p.add_argument('--num_bench', type=int, default=20, help='Benchmark iterations for timing (default: 20, same as testmodel.py)')
    p.add_argument('--num_frames', type=int, default=_T_DEFAULT, help=f'Temporal window T (default: {_T_DEFAULT})')
    p.add_argument('--output', type=str, default='scripts/benchmark_results.csv', help='CSV output path (default: scripts/benchmark_results.csv)')
    p.add_argument('--skip_peakmem', action='store_true', help='Skip Peak Memory @4K measurement')
    p.add_argument('--skip_fps', action='store_true', help='Skip FPS measurements at 720p/1080p/2K/2.7K')
    return p.parse_args()

def main() -> None:
    args = parse_args()
    all_names = list(_MODEL_REGISTRY.keys())
    if args.models is not None:
        unknown = [n for n in args.models if n not in _MODEL_REGISTRY]
        if unknown:
            print(f'[ERROR] Unknown model names: {unknown}')
            print(f'Available: {all_names}')
            sys.exit(1)
        selected = args.models
    else:
        selected = all_names
    if args.skip_peakmem and args.skip_fps:
        print('[ERROR] --skip_peakmem and --skip_fps both set; nothing to do.')
        sys.exit(1)
    print('=' * 70)
    print('  Benchmark: Peak Mem @4K + FPS @720p/1080p/2K/2.7K')
    print('=' * 70)
    print(f'Conditions: B=1, T={args.num_frames}, FP16 autocast, CUDA Event timing')
    print(f'            warmup={args.num_warmup}, bench={args.num_bench}')
    print('=' * 70)
    print()
    (rows, col_headers) = run_benchmark(model_names=selected, num_frames=args.num_frames, num_warmup=args.num_warmup, num_bench=args.num_bench, skip_peakmem=args.skip_peakmem, skip_fps=args.skip_fps, output_path=Path(args.output))
    print('=' * 70)
    print('  Results')
    print('=' * 70)
    print()
    print_markdown_table(rows, col_headers)
    output_path = Path(args.output)
    save_csv(rows, col_headers, output_path)
    print()
    print('=' * 70)
    print('  Done.')
    if _CUDA:
        print(f'  Session peak VRAM: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB')
    print('=' * 70)
if __name__ == '__main__':
    main()
