import argparse
import gc
import math
import statistics
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
_CUDA = torch.cuda.is_available()
_W = 78
_SEP = '=' * _W
_LINE = '-' * _W

def _sync() -> None:
    if _CUDA:
        torch.cuda.synchronize()

def _peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1024 ** 2 if _CUDA else 0.0

def _reset_peak() -> None:
    if _CUDA:
        torch.cuda.reset_peak_memory_stats()

def _bench(fn, nw: int=5, nb: int=20) -> dict[str, float]:
    for _ in range(nw):
        fn()
    if _CUDA:
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)
        ts: list[float] = []
        for _ in range(nb):
            torch.cuda.synchronize()
            ev_s.record()
            fn()
            ev_e.record()
            torch.cuda.synchronize()
            ts.append(ev_s.elapsed_time(ev_e))
    else:
        ts = []
        for _ in range(nb):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000.0)
    return {'mean': statistics.mean(ts), 'std': statistics.stdev(ts) if len(ts) > 1 else 0.0, 'min': min(ts), 'max': max(ts)}

def _header(idx: int | str, title: str) -> None:
    print(f'\n{_SEP}')
    print(f'  [{idx}] {title}')
    print(_SEP)

def _trow(cols: list[str], ws: list[int]) -> str:
    return '  '.join((f'{c:<{w}}' for (c, w) in zip(cols, ws)))
_RES: dict[int, tuple[int, int]] = {256: (256, 256), 512: (512, 512), 720: (1280, 720), 1080: (1920, 1080), 2160: (3840, 2160)}

def _params_m(module: nn.Module) -> float:
    return sum((p.numel() for p in module.parameters())) / 1000000.0

def _resolve_model_config(config_path: str | None) -> tuple[dict | None, str, bool]:
    if config_path is not None:
        with open(config_path) as f:
            return (yaml.safe_load(f), config_path, True)
    fallback_path = 'configs/revide.yaml'
    try:
        with open(fallback_path) as f:
            cfg = yaml.safe_load(f)
        return (cfg, f'(none, auto-loaded model defaults from {fallback_path})', False)
    except FileNotFoundError:
        return (None, '(none, using constructor defaults)', False)

def section_overview(model: nn.Module, device: torch.device, T: int) -> None:
    _header(0, 'Model Overview')
    raw = model.module if hasattr(model, 'module') else model
    enc_s = raw.encode_size
    print(f'  encode_size:  {enc_s}x{enc_s}')
    print(f'  num_frames:   {T}')
    print(f'  Grid spatial: {enc_s // 16}x{enc_s // 16} (backbone stride 16)')
    print(f'  SC grid:      [B, 12, {raw.sc_net.color_bins}, {enc_s // 16}, {enc_s // 16}]')
    print(f'  T grid:       [B, 12, {T}, {enc_s // 16}, {enc_s // 16}]')
    modules = [('SC Net (Spatial-Color)', raw.sc_net), ('T Net (Temporal)', raw.t_net), ('Grid Slicer', raw.slicer), ('DRN (Detail Recovery)', raw.detail_recovery)]
    total_p = _params_m(raw)
    ws = [28, 14, 10]
    print()
    print(_trow(['Module', 'Params (M)', 'Ratio (%)'], ws))
    print(f'  {_LINE}')
    for (name, mod) in modules:
        p = _params_m(mod)
        print(_trow([name, f'{p:.4f}', f'{p / total_p * 100:.1f}%'], ws))
    print(f"  {'─' * 28}  {'─' * 14}  {'─' * 10}")
    print(_trow(['Total', f'{total_p:.4f}', '100.0%'], ws))
    try:
        from torch.utils.flop_counter import FlopCounterMode
        x_flop = torch.randn(1, T, 3, 512, 512, device=device)
        raw.eval()
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            raw(x_flop)
        gflops = fc.get_total_flops() / 1000000000.0
        print(f'\n  FLOPs (forward, 512x512 input): {gflops:.2f} GFLOPs')
        del x_flop
    except Exception as e:
        print(f'\n  FLOPs estimation failed: {e}')
    gc.collect()
    if _CUDA:
        torch.cuda.empty_cache()

def section_latency(model: nn.Module, device: torch.device, T: int, resolutions: list[int], nw: int, nb: int, crop: int) -> None:
    _header(1, 'Latency & Throughput (synthetic data, no I/O)')
    ws = [20, 14, 4, 12, 10, 10, 12]
    print(_trow(['Mode', 'Resolution', 'FP', 'Latency (ms)', 'Std (ms)', 'FPS', 'VRAM (MB)'], ws))
    print(f'  {_LINE}')
    for (fp, amp) in [(32, False), (16, True)]:
        x = torch.randn(1, T, 3, crop, crop, device=device)
        _reset_peak()

        def _fn(_x=x, _amp=amp) -> None:
            with torch.amp.autocast('cuda', enabled=_amp and _CUDA):
                model.train()
                model(_x)
            _sync()
        try:
            s = _bench(_fn, nw, nb)
            vram = _peak_mb()
            fps = 1000.0 / s['mean'] if s['mean'] > 0 else float('inf')
            print(_trow(['train', f'{crop}x{crop}', str(fp), f"{s['mean']:.2f}", f"{s['std']:.2f}", f'{fps:.1f}', f'{vram:.0f}'], ws))
        except RuntimeError as e:
            print(f'  train {crop}x{crop} FP{fp}: OOM — {e}')
        finally:
            del x
            gc.collect()
            if _CUDA:
                torch.cuda.empty_cache()
    print()
    for rk in resolutions:
        if rk not in _RES:
            print(f'  [skip] unknown resolution key {rk}')
            continue
        (W, H) = _RES[rk]
        for (fp, amp) in [(32, False), (16, True)]:
            x = torch.randn(1, T, 3, H, W, device=device)
            _reset_peak()

            def _fn(_x=x, _h=H, _w=W, _amp=amp) -> None:
                model.eval()
                with torch.no_grad(), torch.amp.autocast('cuda', enabled=_amp and _CUDA):
                    model.inference(_x, target_h=_h, target_w=_w)
                _sync()
            try:
                s = _bench(_fn, nw, nb)
                vram = _peak_mb()
                fps = 1000.0 / s['mean'] if s['mean'] > 0 else float('inf')
                print(_trow(['inference', f'{W}x{H}', str(fp), f"{s['mean']:.2f}", f"{s['std']:.2f}", f'{fps:.1f}', f'{vram:.0f}'], ws))
            except RuntimeError as e:
                print(f'  inference {W}x{H} FP{fp}: OOM — {e}')
            finally:
                del x
                gc.collect()
                if _CUDA:
                    torch.cuda.empty_cache()

def section_breakdown(model: nn.Module, device: torch.device, T: int, nw: int, nb: int, input_h: int=512, input_w: int=512) -> None:
    raw = model.module if hasattr(model, 'module') else model
    enc_s = raw.encode_size
    _header(2, f'Module Breakdown (input={input_w}x{input_h}, FP16)')
    model.eval()
    (B, C) = (1, 3)
    x_orig = torch.randn(B, T, C, input_h, input_w, device=device)
    center_hazy = x_orig[:, T // 2]
    all_frames = x_orig.reshape(B * T, C, input_h, input_w)
    all_frames_lr = F.interpolate(all_frames, size=(enc_s, enc_s), mode='bilinear', align_corners=False).reshape(B, T, C, enc_s, enc_s)
    center_hazy_lr = all_frames_lr[:, T // 2]
    p = raw.detail_unshuffle
    pad_h = (p - input_h % p) % p
    pad_w = (p - input_w % p) % p
    (mid_h, mid_w) = ((input_h + pad_h) // p, (input_w + pad_w) // p)
    center_hazy_mid = F.interpolate(center_hazy, size=(mid_h, mid_w), mode='bilinear', align_corners=False)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=_CUDA):
        (sc_grid_r, sc_grid_g) = raw.sc_net(center_hazy_lr)
        t_grid = raw.t_net(all_frames_lr)
        lr_output = raw.slicer.slice_and_apply(sc_grid_r, sc_grid_g, t_grid, center_hazy_mid, T // 2, T)
    rows: list[tuple[str, float, float]] = []

    def _add(name: str, mod: nn.Module | None, fn) -> None:
        s = _bench(fn, nw, nb)
        p = _params_m(mod) if mod is not None else 0.0
        rows.append((name, s['mean'], p))

    def _down(_x=x_orig) -> None:
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=_CUDA):
            _all = _x.reshape(B * T, C, input_h, input_w)
            F.interpolate(_all, size=(enc_s, enc_s), mode='bilinear', align_corners=False)
        _sync()
    _add(f'Bilinear down ({input_w}x{input_h}->{enc_s})', None, _down)

    def _sc(_inp=center_hazy_lr) -> None:
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=_CUDA):
            raw.sc_net(_inp)
        _sync()
    _add('SC Net (center frame)', raw.sc_net, _sc)

    def _tn(_inp=all_frames_lr) -> None:
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=_CUDA):
            raw.t_net(_inp)
        _sync()
    _add(f'T Net ({T} frames)', raw.t_net, _tn)

    def _slicer(_scr=sc_grid_r, _scg=sc_grid_g, _tg=t_grid, _cmid=center_hazy_mid) -> None:
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=_CUDA):
            raw.slicer.slice_and_apply(_scr, _scg, _tg, _cmid, T // 2, T)
        _sync()
    _add(f'Grid Slicer (@{mid_w}x{mid_h})', raw.slicer, _slicer)

    def _drn(_lr=lr_output, _hr=center_hazy, _h=input_h, _w=input_w) -> None:
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=_CUDA):
            raw.detail_recovery(_lr, _hr, _h, _w)
        _sync()
    _add(f'DRN (->{input_w}x{input_h})', raw.detail_recovery, _drn)
    total_ms = sum((ms for (_, ms, _) in rows))
    total_p = sum((p for (_, _, p) in rows))
    ws = [36, 12, 12, 10]
    print(_trow(['Module', 'Latency (ms)', 'Params (M)', 'Ratio (%)'], ws))
    print(f'  {_LINE}')
    for (name, ms, p) in rows:
        ratio = ms / total_ms * 100 if total_ms > 0 else 0
        print(_trow([name, f'{ms:.2f}', f'{p:.4f}' if p > 0 else '-', f'{ratio:.1f}%'], ws))
    print(f"  {'─' * 36}  {'─' * 12}  {'─' * 12}  {'─' * 10}")
    print(_trow(['Total', f'{total_ms:.2f}', f'{total_p:.4f}', '100.0%'], ws))
    print(f'\n  -- DRN latency vs. output resolution --')
    ws2 = [20, 12, 12]
    print(_trow(['Resolution', 'Latency (ms)', 'VRAM (MB)'], ws2))
    for rk in [256, 512, 720, 1080]:
        if rk not in _RES:
            continue
        (dw, dh) = _RES[rk]
        hr = torch.randn(B, C, dh, dw, device=device)
        _reset_peak()

        def _drn_r(_lr=lr_output, _hr=hr, _h=dh, _w=dw) -> None:
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=_CUDA):
                raw.detail_recovery(_lr, _hr, _h, _w)
            _sync()
        try:
            s = _bench(_drn_r, nw, nb)
            print(_trow([f'{dw}x{dh}', f"{s['mean']:.2f}", f'{_peak_mb():.0f}'], ws2))
        except RuntimeError:
            print(_trow([f'{dw}x{dh}', 'OOM', '-'], ws2))
        finally:
            del hr
            gc.collect()
            if _CUDA:
                torch.cuda.empty_cache()
    del x_orig, center_hazy, all_frames_lr, center_hazy_lr
    del sc_grid_r, sc_grid_g, t_grid, lr_output
    gc.collect()
    if _CUDA:
        torch.cuda.empty_cache()

def section_dataloader(cfg: dict, num_batches: int=50) -> None:
    _header(3, 'DataLoader Speed')
    from data import VideoDehazeDataset
    data_cfg = cfg['data']
    split = data_cfg.get('test_split', 'Test')
    try:
        ds = VideoDehazeDataset(root=data_cfg['dataset_root'], split=split, num_frames=data_cfg['num_frames'], crop_size=0)
    except FileNotFoundError as e:
        print(f'  [skip] dataset not found: {e}')
        return
    if len(ds) == 0:
        print('  [skip] dataset is empty')
        return
    nw = int(data_cfg.get('test_num_workers', data_cfg.get('num_workers', 0)))
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=nw, pin_memory=data_cfg.get('pin_memory', True))
    actual = min(num_batches, len(loader))
    times: list[float] = []
    t0 = time.perf_counter()
    for (i, _) in enumerate(loader):
        t1 = time.perf_counter()
        if i > 0:
            times.append((t1 - t0) * 1000.0)
        if i >= actual:
            break
        t0 = t1
    if not times:
        print('  [skip] not enough batches')
        return
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    sample = ds[0]
    print(f'  split:        {split}')
    print(f'  num_workers:  {nw}')
    print(f'  batches:      {len(times)}')
    print(f"  sample shape: hazy={list(sample['hazy_frames'].shape)}, clear={list(sample['clear'].shape)}")
    print(f'  Latency: {mean:.2f} ms/batch  Std: {std:.2f} ms  Throughput: {1000.0 / mean:.1f} FPS')

def section_metrics(device: torch.device, nw: int, nb: int, H: int=512, W: int=512) -> None:
    _header(4, f'Metric Overhead ({W}x{H})')
    from utils import calc_psnr, calc_ssim, calc_psnr_torch, calc_ssim_torch, tensor_to_uint8
    pred_t = torch.rand(3, H, W, device=device)
    gt_t = torch.rand(3, H, W, device=device)
    pred_np = (pred_t.cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    gt_np = (gt_t.cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    items: list[tuple[str, float]] = []

    def _add(name: str, fn) -> None:
        s = _bench(fn, nw, nb)
        items.append((name, s['mean']))
    _add('tensor_to_uint8', lambda : tensor_to_uint8(pred_t))
    _add('PSNR-CPU (skimage)', lambda : calc_psnr(pred_np, gt_np))
    _add('SSIM-CPU (skimage)', lambda : calc_ssim(pred_np, gt_np))

    def _pg() -> None:
        calc_psnr_torch(pred_t, gt_t)
        _sync()
    _add('PSNR-GPU (torch)', _pg)

    def _sg() -> None:
        calc_ssim_torch(pred_t, gt_t)
        _sync()
    _add('SSIM-GPU (torch)', _sg)
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
        (p2, g2) = (pred_t.unsqueeze(0) * 2 - 1, gt_t.unsqueeze(0) * 2 - 1)

        def _lp() -> None:
            with torch.no_grad():
                lpips_fn(p2, g2)
            _sync()
        _add('LPIPS (AlexNet)', _lp)
    except ImportError:
        items.append(('LPIPS (AlexNet)', float('nan')))

    def _cpu_all() -> None:
        pnp = tensor_to_uint8(pred_t)
        gnp = tensor_to_uint8(gt_t)
        calc_psnr(pnp, gnp)
        calc_ssim(pnp, gnp)
    _add('Combined-CPU (test.py)', _cpu_all)

    def _gpu_all() -> None:
        calc_psnr_torch(pred_t, gt_t)
        calc_ssim_torch(pred_t, gt_t)
        _sync()
    _add('Combined-GPU (train.py)', _gpu_all)
    ws = [28, 14]
    print(_trow(['Metric', 'Latency (ms)'], ws))
    print(f'  {_LINE}')
    for (name, ms) in items:
        val = f'{ms:.3f}' if not math.isnan(ms) else 'N/A'
        print(_trow([name, val], ws))
    cpu_ms = next((ms for (n, ms) in items if 'Combined-CPU' in n), 0)
    gpu_ms = next((ms for (n, ms) in items if 'Combined-GPU' in n), 0)
    if cpu_ms > 0 and gpu_ms > 0:
        print(f'\n  CPU metrics are {cpu_ms / gpu_ms:.1f}x slower than GPU per frame')

def section_pipeline(model: nn.Module, cfg: dict, device: torch.device, nw: int, nb: int) -> None:
    _header(5, 'Full Pipeline (real data)')
    from data import VideoDehazeDataset
    from utils import calc_psnr, calc_ssim, calc_psnr_torch, calc_ssim_torch, tensor_to_uint8
    data_cfg = cfg['data']
    split = data_cfg.get('test_split', 'Test')
    try:
        ds = VideoDehazeDataset(root=data_cfg['dataset_root'], split=split, num_frames=data_cfg['num_frames'], crop_size=0)
    except FileNotFoundError as e:
        print(f'  [skip] dataset not found: {e}')
        return
    if len(ds) == 0:
        print('  [skip] dataset is empty')
        return
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    raw = model.module if hasattr(model, 'module') else model
    n_total = min(nw + nb, len(loader))
    t_inf: list[float] = []
    t_cpu: list[float] = []
    t_gpu: list[float] = []
    for (i, batch) in enumerate(loader):
        if i >= n_total:
            break
        warmup = i < nw
        hazy = batch['hazy_frames'].to(device, non_blocking=True)
        clear = batch['clear'].to(device, non_blocking=True)
        (_, _, _, th, tw) = hazy.shape
        _sync()
        if _CUDA:
            es = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            es.record()
        else:
            _t0 = time.perf_counter()
        with torch.no_grad():
            output = raw.inference(hazy, target_h=th, target_w=tw)
        if _CUDA:
            ee.record()
            torch.cuda.synchronize()
            ms_inf = es.elapsed_time(ee)
        else:
            ms_inf = (time.perf_counter() - _t0) * 1000.0
        tc = time.perf_counter()
        pnp = tensor_to_uint8(output[0])
        gnp = tensor_to_uint8(clear[0])
        calc_psnr(pnp, gnp)
        calc_ssim(pnp, gnp)
        ms_cpu = (time.perf_counter() - tc) * 1000.0
        _sync()
        tg = time.perf_counter()
        calc_psnr_torch(output[0], clear[0])
        calc_ssim_torch(output[0], clear[0])
        _sync()
        ms_gpu = (time.perf_counter() - tg) * 1000.0
        if not warmup:
            t_inf.append(ms_inf)
            t_cpu.append(ms_cpu)
            t_gpu.append(ms_gpu)
    if not t_inf:
        print('  [skip] not enough samples')
        return

    def _s(lst: list[float]) -> tuple[float, float]:
        return (statistics.mean(lst), statistics.stdev(lst) if len(lst) > 1 else 0.0)
    (im, isd) = _s(t_inf)
    (cm, csd) = _s(t_cpu)
    (gm, gsd) = _s(t_gpu)
    tot_cpu = im + cm
    tot_gpu = im + gm
    ws = [36, 12, 10, 10]
    print(f'  Resolution: {th}x{tw}  Samples: {len(t_inf)}')
    print()
    print(_trow(['Component', 'Latency (ms)', 'Std (ms)', 'Ratio (%)'], ws))
    print(f'  {_LINE}')
    print(_trow(['model.inference()', f'{im:.2f}', f'{isd:.2f}', f'{im / tot_cpu * 100:.1f}%'], ws))
    print(_trow(['+ CPU metrics (test.py)', f'{cm:.2f}', f'{csd:.2f}', f'{cm / tot_cpu * 100:.1f}%'], ws))
    print(_trow(['  (alt) GPU metrics', f'{gm:.2f}', f'{gsd:.2f}', '(alt)'], ws))
    print(f"  {'─' * 36}  {'─' * 12}  {'─' * 10}  {'─' * 10}")
    print(_trow(['Total (inference + CPU metrics)', f'{tot_cpu:.2f}', '', '100.0%'], ws))
    fps_pure = 1000.0 / im if im > 0 else float('inf')
    fps_cpu = 1000.0 / tot_cpu if tot_cpu > 0 else float('inf')
    fps_gpu = 1000.0 / tot_gpu if tot_gpu > 0 else float('inf')
    print(f'\n  FPS Summary:')
    print(f'    Pure inference:            {fps_pure:.2f} FPS')
    print(f'    + CPU metrics (test.py):   {fps_cpu:.2f} FPS')
    print(f'    + GPU metrics (train.py):  {fps_gpu:.2f} FPS')
    print(f'    Metric overhead:           {cm / tot_cpu * 100:.1f}% (CPU) / {gm / tot_gpu * 100:.1f}% (GPU)')

def _build_model(cfg: dict | None, checkpoint: str | None, device: torch.device, explicit_config: bool) -> nn.Module:
    from models import BilateralGridDehazeNet
    if cfg is not None:
        mc = cfg['model']
        model = BilateralGridDehazeNet(backbone_channels=mc.get('backbone_channels', [24, 48, 96, 192]), color_bins=mc.get('color_bins', 8), coeff_dim=mc.get('coeff_dim', 12), num_frames=mc.get('temporal_num_frames', 5), encode_size=mc.get('encode_size', 256), detail_unshuffle=mc.get('detail_unshuffle', 4), detail_channels=mc.get('detail_channels', 64), drop_path_rate=mc.get('drop_path_rate', 0.1), grid_dropout=mc.get('grid_dropout', 0.1), temporal_attn_heads=mc.get('temporal_attn_heads', 4))
        build_source = 'config-loaded' if explicit_config else 'auto-loaded model config'
    else:
        model = BilateralGridDehazeNet()
        build_source = 'default constructor'
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        sd = ckpt['model'] if 'model' in ckpt else ckpt
        model.load_state_dict(sd)
        print(f'  Checkpoint: {checkpoint}')
    print(f'  Build:      {build_source}')
    if cfg is None:
        print('  Note: default-constructor params do not represent training config.')
    elif not explicit_config:
        print('  Note: data sections still require an explicit --config.')
    model.to(device).eval()
    print(f'  Model: BilateralGridDehazeNet — {_params_m(model):.2f}M params on {device}')
    return model

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='BilateralGridDehazeNet evaluation: params, FLOPs, latency, memory', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='Examples:\n  python testmodel.py                                   # model-only\n  python testmodel.py --config configs/revide.yaml      # full diagnosis\n  python testmodel.py --sections 0 2 4                  # overview + breakdown + metrics\n')
    p.add_argument('--config', type=str, default=None, help='Config yaml (enables DataLoader & pipeline tests)')
    p.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path (random weights if omitted)')
    p.add_argument('--num_warmup', type=int, default=5, help='Warmup iterations per benchmark')
    p.add_argument('--num_bench', type=int, default=20, help='Benchmark iterations per test')
    p.add_argument('--num_frames', type=int, default=5, help='Temporal window T')
    p.add_argument('--train_crop', type=int, default=512, help='Training crop size')
    p.add_argument('--resolutions', type=int, nargs='+', default=[256, 512, 720, 1080, 2160], help='Inference resolution keys (256/512/720/1080/2160)')
    p.add_argument('--sections', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5], help='Sections to run (0-5)')
    return p.parse_args()

def main() -> None:
    args = parse_args()
    explicit_cfg: dict | None = None
    if args.config is not None:
        with open(args.config) as f:
            explicit_cfg = yaml.safe_load(f)
    (model_cfg, config_desc, explicit_model_config) = _resolve_model_config(args.config)
    device = torch.device('cuda' if _CUDA else 'cpu')
    print(_SEP)
    print('  BilateralGridDehazeNet — Model Evaluation Report')
    print(_SEP)
    print(f'  Device:     {device}')
    if _CUDA:
        props = torch.cuda.get_device_properties(0)
        print(f'  GPU:        {torch.cuda.get_device_name(0)}')
        print(f'  VRAM:       {props.total_memory / 1024 ** 3:.1f} GB')
        print(f'  PyTorch:    {torch.__version__}')
    print(f'  Warmup/Bench: {args.num_warmup} / {args.num_bench}')
    print(f'  Config:     {config_desc}')
    print(f"  Checkpoint: {args.checkpoint or '(random weights)'}")
    model = _build_model(model_cfg, args.checkpoint, device, explicit_model_config)
    T = args.num_frames
    if model_cfg is not None:
        T = model_cfg['data'].get('num_frames', T)
    crop = args.train_crop
    if model_cfg is not None:
        crop = model_cfg['data'].get('train_crop_size', crop)
    sections = set(args.sections)
    if 0 in sections:
        section_overview(model, device, T)
    if 1 in sections:
        section_latency(model, device, T, args.resolutions, args.num_warmup, args.num_bench, crop)
    if 2 in sections:
        section_breakdown(model, device, T, args.num_warmup, args.num_bench, crop, crop)
    if 3 in sections:
        if explicit_cfg is not None:
            section_dataloader(explicit_cfg, num_batches=args.num_warmup + args.num_bench)
        else:
            _header(3, 'DataLoader Speed')
            print('  [skip] --config not provided')
    if 4 in sections:
        section_metrics(device, args.num_warmup, args.num_bench)
    if 5 in sections:
        if explicit_cfg is not None:
            section_pipeline(model, explicit_cfg, device, args.num_warmup, args.num_bench)
        else:
            _header(5, 'Full Pipeline')
            print('  [skip] --config not provided')
    print(f'\n{_SEP}')
    print('  Evaluation complete.')
    if _CUDA:
        print(f'  Peak VRAM: {torch.cuda.max_memory_allocated() / 1024 ** 2:.0f} MB')
    print(_SEP)
if __name__ == '__main__':
    main()
