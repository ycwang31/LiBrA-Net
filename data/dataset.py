import random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

def _resolve_split_dir(root: Path, split: str) -> Path:
    for name in (split, split.capitalize(), split.lower(), split.upper()):
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Split "{split}" not found under {root}. Please configure train_split / val_split / test_split explicitly.')

class VideoDehazeDataset(Dataset):

    def __init__(self, root: str, split: str='train', num_frames: int=5, crop_size: int=512) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.is_train = split.lower() in ('train',)
        self.split_dir = _resolve_split_dir(self.root, split)
        hazy_root = self.split_dir / 'hazy'
        gt_root = self.split_dir / 'gt'
        self.samples: list[dict] = []
        if hazy_root.exists() and gt_root.exists():
            for video_dir in sorted(hazy_root.iterdir()):
                if not video_dir.is_dir():
                    continue
                video_name = video_dir.name
                if not (gt_root / video_name).exists():
                    continue
                frames = sorted((p for p in video_dir.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')))
                n_frames = len(frames)
                if n_frames == 0:
                    continue
                half_t = num_frames // 2
                ext = frames[0].suffix
                for center_idx in range(half_t, n_frames - half_t):
                    self.samples.append({'video_name': video_name, 'center_idx': center_idx, 'n_frames': n_frames, 'ext': ext})

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str | Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0

    def _random_crop_params(self, h: int, w: int) -> tuple[int, int]:
        top = random.randint(0, max(0, h - self.crop_size))
        left = random.randint(0, max(0, w - self.crop_size))
        return (top, left)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        video_name = sample['video_name']
        center_idx = sample['center_idx']
        half_t = self.num_frames // 2
        ext = sample['ext']
        hazy_frames = []
        for t in range(center_idx - half_t, center_idx + half_t + 1):
            path = self.split_dir / 'hazy' / video_name / f'{t:05d}{ext}'
            hazy_frames.append(self._load_image(path))
        clear = self._load_image(self.split_dir / 'gt' / video_name / f'{center_idx:05d}{ext}')
        (h, w) = clear.shape[:2]
        if self.is_train:
            (top, left) = self._random_crop_params(h, w)
            cs = self.crop_size
            hazy_frames = [f[top:top + cs, left:left + cs] for f in hazy_frames]
            clear = clear[top:top + cs, left:left + cs]
            if random.random() > 0.5:
                hazy_frames = [f[:, ::-1].copy() for f in hazy_frames]
                clear = clear[:, ::-1].copy()
            if random.random() > 0.5:
                hazy_frames = [f[::-1].copy() for f in hazy_frames]
                clear = clear[::-1].copy()
        hazy_tensor = torch.stack([torch.from_numpy(f.copy()).permute(2, 0, 1) for f in hazy_frames])
        clear_tensor = torch.from_numpy(clear.copy()).permute(2, 0, 1)
        return {'hazy_frames': hazy_tensor, 'clear': clear_tensor, 'video_name': video_name, 'frame_idx': center_idx}
