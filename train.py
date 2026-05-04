import argparse
import importlib
import logging
import os
import time
import warnings
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from data import VideoDehazeDataset
from losses import DehazeLoss
from utils import AverageMeter, all_reduce_scalar_dict, build_lpips_metric, calc_psnr, calc_ssim, evaluate_dehazing_model, get_cosine_schedule_with_warmup, is_main_process, reduce_tensor_dict, set_seed, setup_distributed, setup_logger, tensor_to_uint8
MODEL_REGISTRY: dict[str, tuple[str, str]] = {'ours': ('models', 'BilateralGridDehazeNet'), 'ours_A1': ('models_ablation', 'AblationA1'), 'ours_A2': ('models_ablation', 'AblationA2'), 'ours_A3': ('models_ablation', 'AblationA3'), 'ours_A4': ('models_ablation', 'AblationA4'), 'ours_A5': ('models_ablation', 'AblationA5'), 'ours_A6': ('models_ablation', 'AblationA6'), 'ours_A7': ('models_ablation', 'AblationA7'), 'ours_A8': ('models_ablation', 'AblationA8'), 'ours_A9': ('models_ablation', 'AblationA9'), 'ours_L1': ('models', 'BilateralGridDehazeNet'), 'ours_L2': ('models', 'BilateralGridDehazeNet'), 'ours_L3': ('models', 'BilateralGridDehazeNet'), 'ours_L4': ('models', 'BilateralGridDehazeNet'), 'ours_L5': ('models', 'BilateralGridDehazeNet'), 'ours_L6': ('models', 'BilateralGridDehazeNet'), 'ours_L7': ('models', 'BilateralGridDehazeNet'), 'ours_L8': ('models', 'BilateralGridDehazeNet')}
_OURS_ARCH_IDENTICAL: frozenset[str] = frozenset({'ours', 'ours_L1', 'ours_L2', 'ours_L3', 'ours_L4', 'ours_L5', 'ours_L6', 'ours_L7', 'ours_L8'})
logger = logging.getLogger('dehazing')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Video Dehazing Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to the config file')
    parser.add_argument('--model', type=str, default='ours', choices=list(MODEL_REGISTRY.keys()), help='Model name')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path for resuming training')
    parser.add_argument('--load_weights', type=str, default=None, help='Load model weights only without restoring optimizer state')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local_rank', type=int, default=0, help='DDP local rank (legacy)')
    return parser.parse_args()

def load_config(path: str) -> dict:
    with open(path) as file:
        return yaml.safe_load(file)

def build_model(cfg: dict, model_name: str='ours') -> nn.Module:
    (module_path, class_name) = MODEL_REGISTRY[model_name]
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    model_cfg = cfg['model']
    return model_class(backbone_channels=model_cfg.get('backbone_channels', [24, 48, 96, 192]), color_bins=model_cfg.get('color_bins', 8), coeff_dim=model_cfg.get('coeff_dim', 12), num_frames=model_cfg.get('temporal_num_frames', 5), encode_size=model_cfg.get('encode_size', 256), detail_unshuffle=model_cfg.get('detail_unshuffle', 4), detail_channels=model_cfg.get('detail_channels', 64), drop_path_rate=model_cfg.get('drop_path_rate', 0.1), grid_dropout=model_cfg.get('grid_dropout', 0.1), temporal_attn_heads=model_cfg.get('temporal_attn_heads', 4))

def _resolve_role_split(cfg: dict, role: str) -> str:
    data_cfg = cfg['data']
    return data_cfg.get(f'{role}_split', role.capitalize())

def build_dataloader(cfg: dict, role: str, distributed: bool, eval_distributed: bool=False) -> tuple[VideoDehazeDataset, DataLoader]:
    data_cfg = cfg['data']
    eval_cfg = cfg.get('eval', {})
    split_name = _resolve_role_split(cfg, role)
    is_train = role == 'train'
    base_dataset = VideoDehazeDataset(root=data_cfg['dataset_root'], split=split_name, num_frames=data_cfg['num_frames'], crop_size=data_cfg['train_crop_size'] if is_train else 0)
    dataset = base_dataset
    max_samples = eval_cfg.get('max_samples')
    if not is_train and max_samples is not None:
        max_samples = min(int(max_samples), len(base_dataset))
        dataset = Subset(base_dataset, list(range(max_samples)))
    use_distributed_sampler = distributed and (is_train or eval_distributed)
    sampler = None
    if use_distributed_sampler:
        sampler = DistributedSampler(dataset, shuffle=is_train, drop_last=is_train)
    worker_key = f'{role}_num_workers'
    num_workers = int(data_cfg.get(worker_key, data_cfg.get('num_workers', 0)))
    loader_kwargs = {'dataset': dataset, 'batch_size': cfg['train']['batch_size'] if is_train else 1, 'shuffle': is_train and sampler is None, 'sampler': sampler, 'num_workers': num_workers, 'pin_memory': data_cfg.get('pin_memory', True), 'drop_last': is_train}
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = data_cfg.get('persistent_workers', False)
        if data_cfg.get('prefetch_factor') is not None:
            loader_kwargs['prefetch_factor'] = data_cfg['prefetch_factor']
    return (base_dataset, DataLoader(**loader_kwargs))

def _resolve_exp_dirs(cfg: dict, model_name: str) -> tuple[Path, Path]:
    ckpt_dir = Path(cfg['train']['checkpoint_dir']) / model_name
    log_dir = Path(cfg['train']['log_dir']) / model_name
    return (ckpt_dir, log_dir)

def _compute_batch_metrics(output: torch.Tensor, clear: torch.Tensor, device: torch.device) -> dict[str, float]:
    metric_sums = {'psnr': 0.0, 'ssim': 0.0}
    batch_size = int(output.shape[0])
    for idx in range(batch_size):
        pred_np = tensor_to_uint8(output[idx])
        gt_np = tensor_to_uint8(clear[idx])
        metric_sums['psnr'] += calc_psnr(pred_np, gt_np)
        metric_sums['ssim'] += calc_ssim(pred_np, gt_np)
    return all_reduce_scalar_dict(metric_sums, device=device, average=False)

def _log_runtime_header(cfg: dict, args: argparse.Namespace, device: torch.device, rank: int, world_size: int, local_rank: int) -> None:
    distributed = world_size > 1
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')
    logger.info('Runtime | rank=%d world_size=%d distributed=%s local_rank=%d device=%s visible_devices=%s', rank, world_size, distributed, local_rank, device, visible_devices)
    logger.info('Training | config=%s model=%s batch_per_gpu=%d effective_batch=%d', args.config, args.model, cfg['train']['batch_size'], cfg['train']['batch_size'] * world_size)
    if not distributed:
        logger.warning('Distributed environment not detected; training will run in single-process mode. Use torchrun --nproc_per_node=8 train.py --config %s to enable 8-GPU DDP.', args.config)

def _format_eval_log(metrics: dict[str, float | int | None]) -> str:
    parts = [f"PSNR: {metrics['psnr']:.4f} dB" if metrics['psnr'] is not None else None, f"SSIM: {metrics['ssim']:.6f}" if metrics['ssim'] is not None else None, f"LPIPS: {metrics['lpips']:.6f}" if metrics['lpips'] is not None else None, f"FPS: {metrics['fps']:.2f}" if metrics['fps'] is not None else None, f"ms/frame: {metrics['ms_per_frame']:.2f}" if metrics['ms_per_frame'] is not None else None, f"frames: {metrics['total_frames']}"]
    return ' | '.join((part for part in parts if part is not None))

def train(cfg: dict, args: argparse.Namespace) -> None:
    (rank, world_size, local_rank) = setup_distributed(cfg['distributed']['backend'])
    distributed = world_size > 1
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    set_seed(args.seed + rank)
    (ckpt_dir, log_dir) = _resolve_exp_dirs(cfg, args.model)
    if is_main_process():
        setup_logger('dehazing', log_dir=log_dir, log_filename='train.log')
    else:
        setup_logger('dehazing')
    _log_runtime_header(cfg, args, device, rank, world_size, local_rank)
    logger.info('Checkpoint dir: %s', ckpt_dir)
    logger.info('Log dir: %s', log_dir)
    model = build_model(cfg, args.model).to(device)
    n_params = sum((p.numel() for p in model.parameters())) / 1000000.0
    logger.info('Model [%s] created — %.2fM params', args.model, n_params)
    if distributed:
        find_unused = cfg['distributed']['find_unused_parameters'] or args.model not in _OURS_ARCH_IDENTICAL
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=find_unused)
    train_cfg = cfg['train']
    val_distributed = bool(train_cfg.get('val_distributed', False))
    lie_w = train_cfg.get('loss_lie_weight', 0.3)
    criterion = DehazeLoss(pixel_weight=train_cfg['loss_pixel_weight'], perc_weight=train_cfg['loss_perc_weight'], lie_weight=lie_w, charbonnier_eps=train_cfg['charbonnier_eps'], lie_temporal_weight=train_cfg.get('lie_temporal_weight', 0.08), lie_smooth_weight=train_cfg.get('lie_smooth_weight', 0.05), lie_identity_weight=train_cfg.get('lie_identity_weight', 0.01), lie_s3_weight=train_cfg.get('lie_s3_weight', 0.02), perc_backbone=train_cfg.get('perc_backbone', 'dinov2')).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])
    (train_dataset, train_loader) = build_dataloader(cfg, 'train', distributed)
    val_loader: DataLoader | None = None
    if not distributed or val_distributed or is_main_process():
        (_, val_loader) = build_dataloader(cfg, 'val', distributed, eval_distributed=val_distributed)
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise RuntimeError('Train loader is empty. Please check the dataset and batch size configuration.')
    total_steps = train_cfg['epochs'] * steps_per_epoch
    warmup_steps = train_cfg['warmup_epochs'] * steps_per_epoch
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Detected call of `lr_scheduler.step')
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_iters=warmup_steps, total_iters=total_steps)
    scaler = GradScaler('cuda', enabled=train_cfg['amp'] and device.type == 'cuda')
    if is_main_process() and val_loader is not None:
        logger.info('Data | train_split=%s (%d samples) | val_split=%s (%d samples)', _resolve_role_split(cfg, 'train'), len(train_dataset), _resolve_role_split(cfg, 'val'), len(val_loader.dataset))
        if _resolve_role_split(cfg, 'val').lower() == _resolve_role_split(cfg, 'test').lower():
            logger.warning('Validation split "%s" matches test split "%s". This is an explicit protocol compromise for this run.', _resolve_role_split(cfg, 'val'), _resolve_role_split(cfg, 'test'))
    start_epoch = 0
    global_step = 0
    best_psnr = float('-inf')
    best_ssim = float('-inf')
    epochs_without_improvement = 0
    early_stopping_patience = int(train_cfg.get('early_stopping_patience', 0))
    if args.resume and args.load_weights:
        raise ValueError('--resume and --load_weights are mutually exclusive; choose one mode only.')
    model_to_load = model.module if distributed else model
    if args.load_weights:
        checkpoint_data = torch.load(args.load_weights, map_location=device, weights_only=False)
        state_dict = checkpoint_data['model'] if isinstance(checkpoint_data, dict) and 'model' in checkpoint_data else checkpoint_data
        model_to_load.load_state_dict(state_dict)
        logger.info('Loaded pretrained weights from %s | optimizer/scheduler/scaler not restored', args.load_weights)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model_to_load.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        if 'epoch' in checkpoint:
            start_epoch = int(checkpoint['epoch'])
            global_step = int(checkpoint.get('global_step', start_epoch * steps_per_epoch))
        else:
            legacy_step = int(checkpoint.get('iteration', 0))
            start_epoch = legacy_step // steps_per_epoch
            global_step = legacy_step
            logger.warning('Resuming from a legacy iter-based checkpoint; training will restart from epoch boundary %d.', start_epoch)
        best_psnr = float(checkpoint.get('best_psnr', best_psnr))
        best_ssim = float(checkpoint.get('best_ssim', best_ssim))
        logger.info('Resumed from %s | start_epoch=%d global_step=%d best_psnr=%.4f best_ssim=%.6f', args.resume, start_epoch, global_step, best_psnr, best_ssim)
    writer = None
    if is_main_process():
        from torch.utils.tensorboard import SummaryWriter
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))
    val_metric_names = [metric.lower() for metric in train_cfg.get('val_metrics', ['psnr', 'ssim', 'fps'])]
    lpips_fn = None
    if val_loader is not None and 'lpips' in val_metric_names:
        lpips_fn = build_lpips_metric(device, logger=logger)
    if start_epoch >= train_cfg['epochs']:
        logger.warning('start_epoch=%d exceeds configured epochs=%d; nothing to do.', start_epoch, train_cfg['epochs'])
    for epoch in range(start_epoch, train_cfg['epochs']):
        model.train()
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        loss_meters = {'total': AverageMeter(), 'pixel': AverageMeter(), 'perc': AverageMeter(), 'lie': AverageMeter()}
        metric_meters = {'psnr': AverageMeter(), 'ssim': AverageMeter(), 'data_time': AverageMeter(), 'iter_time': AverageMeter()}
        iterator = enumerate(train_loader, start=1)
        if is_main_process():
            iterator = tqdm(iterator, total=steps_per_epoch, desc=f"Epoch [{epoch + 1}/{train_cfg['epochs']}]", leave=False)
        end_time = time.perf_counter()
        for (step, batch) in iterator:
            data_time = time.perf_counter() - end_time
            metric_meters['data_time'].update(data_time)
            step_start = time.perf_counter()
            hazy_frames = batch['hazy_frames'].to(device, non_blocking=True)
            clear = batch['clear'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=train_cfg['amp'] and device.type == 'cuda'):
                output = model(hazy_frames)
                losses = criterion(output, clear)
                total_loss = losses['total']
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip_max_norm'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            reduced_losses = reduce_tensor_dict({key: value.detach() for (key, value) in losses.items()}, average=True)
            batch_metric_sums = _compute_batch_metrics(output['output'].detach(), clear, device)
            global_batch_size = hazy_frames.shape[0] * world_size
            batch_psnr = batch_metric_sums['psnr'] / max(global_batch_size, 1)
            batch_ssim = batch_metric_sums['ssim'] / max(global_batch_size, 1)
            for (key, meter) in loss_meters.items():
                meter.update(reduced_losses[key].item(), n=global_batch_size)
            metric_meters['psnr'].update(batch_psnr, n=global_batch_size)
            metric_meters['ssim'].update(batch_ssim, n=global_batch_size)
            iter_time = time.perf_counter() - step_start
            metric_meters['iter_time'].update(iter_time)
            end_time = time.perf_counter()
            if is_main_process():
                iterator.set_postfix(step=step, global_step=global_step, lr=f"{optimizer.param_groups[0]['lr']:.2e}", loss=f"{loss_meters['total'].avg:.4f}", psnr=f"{metric_meters['psnr'].avg:.2f}", ssim=f"{metric_meters['ssim'].avg:.4f}", data=f"{metric_meters['data_time'].avg:.3f}s", iter_t=f"{metric_meters['iter_time'].avg:.3f}s")
            if is_main_process() and (step % train_cfg['log_every_steps'] == 0 or step == steps_per_epoch):
                lr = optimizer.param_groups[0]['lr']
                logger.info('[Epoch %d/%d][Iter %d/%d][Step %d] Loss: %.4f PSNR: %.2f SSIM: %.4f LR: %.6f (pix=%.3f perc=%.3f lie=%.3f data=%.3fs iter=%.3fs)', epoch + 1, train_cfg['epochs'], step, steps_per_epoch, global_step, loss_meters['total'].avg, metric_meters['psnr'].avg, metric_meters['ssim'].avg, lr, loss_meters['pixel'].avg, loss_meters['perc'].avg, loss_meters['lie'].avg, metric_meters['data_time'].avg, metric_meters['iter_time'].avg)
                if writer is not None:
                    writer.add_scalar('train/loss_total', loss_meters['total'].avg, global_step)
                    writer.add_scalar('train/loss_pixel', loss_meters['pixel'].avg, global_step)
                    writer.add_scalar('train/loss_perc', loss_meters['perc'].avg, global_step)
                    writer.add_scalar('train/loss_lie', loss_meters['lie'].avg, global_step)
                    writer.add_scalar('train/psnr', metric_meters['psnr'].avg, global_step)
                    writer.add_scalar('train/ssim', metric_meters['ssim'].avg, global_step)
                    writer.add_scalar('train/lr', lr, global_step)
                    writer.add_scalar('train/data_time', metric_meters['data_time'].avg, global_step)
                    writer.add_scalar('train/iter_time', metric_meters['iter_time'].avg, global_step)
                for meter in list(loss_meters.values()) + list(metric_meters.values()):
                    meter.reset()
        should_validate = (epoch + 1) % train_cfg['val_every_epochs'] == 0
        if should_validate:
            model.eval()
            val_results = None
            if val_loader is not None:
                val_results = evaluate_dehazing_model(model=model, data_loader=val_loader, device=device, metric_names=val_metric_names, distributed=distributed and val_distributed, lpips_fn=lpips_fn, save_dir=None, show_progress=is_main_process(), description=f'Val Epoch {epoch + 1}', amp_inference=True, gpu_metrics=True)
            if is_main_process() and val_results is not None:
                logger.info('[Val][%s][Epoch %d/%d][Step %d] %s', 'ddp-sharded' if distributed and val_distributed else 'single-rank rank0', epoch + 1, train_cfg['epochs'], global_step, _format_eval_log(val_results))
                if writer is not None:
                    if val_results['psnr'] is not None:
                        writer.add_scalar('val/psnr', val_results['psnr'], global_step)
                    if val_results['ssim'] is not None:
                        writer.add_scalar('val/ssim', val_results['ssim'], global_step)
                    if val_results['lpips'] is not None:
                        writer.add_scalar('val/lpips', val_results['lpips'], global_step)
                    if val_results['fps'] is not None:
                        writer.add_scalar('val/fps', val_results['fps'], global_step)
                    if val_results['ms_per_frame'] is not None:
                        writer.add_scalar('val/ms_per_frame', val_results['ms_per_frame'], global_step)
                val_psnr = float(val_results['psnr']) if val_results['psnr'] is not None else float('-inf')
                val_ssim = float(val_results['ssim']) if val_results['ssim'] is not None else float('-inf')
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    best_ssim = val_ssim
                    epochs_without_improvement = 0
                    model_to_save = model.module if distributed else model
                    best_checkpoint = {'model': model_to_save.state_dict(), 'epoch': epoch + 1, 'global_step': global_step, 'config': cfg, 'model_name': args.model, 'best_psnr': best_psnr, 'best_ssim': best_ssim}
                    torch.save(best_checkpoint, ckpt_dir / 'best.pth')
                    logger.info('Best model updated (PSNR=%.4f, SSIM=%.6f) → %s', best_psnr, best_ssim, ckpt_dir / 'best.pth')
                else:
                    epochs_without_improvement += train_cfg['val_every_epochs']
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                if is_main_process():
                    logger.info('Early stopping triggered: no improvement for %d epochs (patience=%d). Best PSNR=%.4f at epoch %d.', epochs_without_improvement, early_stopping_patience, best_psnr, epoch + 1 - epochs_without_improvement)
                break
            if distributed:
                dist.barrier()
            model.train()
        should_save = (epoch + 1) % train_cfg['save_every_epochs'] == 0
        if is_main_process() and should_save:
            model_to_save = model.module if distributed else model
            checkpoint = {'model': model_to_save.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch + 1, 'global_step': global_step, 'config': cfg, 'model_name': args.model, 'best_psnr': best_psnr, 'best_ssim': best_ssim}
            save_path = ckpt_dir / f'epoch_{epoch + 1:04d}.pth'
            torch.save(checkpoint, save_path)
            logger.info('Checkpoint saved → %s', save_path)
    if is_main_process():
        model_to_save = model.module if distributed else model
        last_checkpoint = {'model': model_to_save.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'epoch': train_cfg['epochs'], 'global_step': global_step, 'config': cfg, 'model_name': args.model, 'best_psnr': best_psnr, 'best_ssim': best_ssim}
        torch.save(last_checkpoint, ckpt_dir / 'last.pth')
        logger.info('Last checkpoint saved → %s', ckpt_dir / 'last.pth')
        logger.info('Training finished | best_psnr=%.4f best_ssim=%.6f', best_psnr, best_ssim)
    if writer is not None:
        writer.close()
    if dist.is_initialized():
        dist.destroy_process_group()
if __name__ == '__main__':
    parsed_args = parse_args()
    config = load_config(parsed_args.config)
    train(config, parsed_args)
