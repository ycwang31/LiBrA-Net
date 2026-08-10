# LiBrA-Net: Lie-Algebraic Bilateral Affine Fields for Real-Time 4K Video Dehazing

LiBrA-Net is a video dehazing framework for native 4K content. It factorizes the spatiotemporal affine field into compact spatial-color and temporal bilateral grids, fuses their transformations in the Lie algebra $\mathfrak{gl}(3)$ through a Cayley parameterization, and restores high-frequency content with an input-guided detail recovery branch.

## UHV-4K Dataset

UHV-4K is a paired 4K video dehazing benchmark containing 100 videos at 3840 × 2160 resolution across seven scene categories. The dataset includes clean and hazy video pairs together with depth, transmission, and optical-flow annotations.

| Resource | Link |
| --- | --- |
| UHV-4K dataset | [Baidu Netdisk (code: `9w57`)](https://pan.baidu.com/s/1Phz1A3r0pxWTRoQEmRq47g?pwd=9w57) |

Arrange each dataset as follows:

```text
dataset_root/
├── Train/
│   ├── hazy/
│   │   ├── sequence_001/
│   │   │   ├── 00000.jpg
│   │   │   ├── 00001.jpg
│   │   │   └── ...
│   │   └── ...
│   └── gt/
│       ├── sequence_001/
│       │   ├── 00000.jpg
│       │   ├── 00001.jpg
│       │   └── ...
│       └── ...
└── Test/
    ├── hazy/
    └── gt/
```

The split names are case-insensitive. Frame names must use zero-padded indices such as `00000.jpg`; `.png`, `.jpg`, and `.jpeg` are supported.

## Requirements

- Python 3.10+
- PyTorch 2.1+
- CUDA-capable GPU

Install the Python dependencies:

```bash
git clone https://github.com/ycwang31/LiBrA-Net.git
cd LiBrA-Net
pip install -r requirements.txt
```

Training uses a local DINOv2 ViT-S/14 backbone for perceptual supervision. Place the DINOv2 repository and weights at the paths expected by `losses.py`:

```bash
mkdir -p pretrained
git clone https://github.com/facebookresearch/dinov2.git pretrained/facebookresearch_dinov2_main
wget -O pretrained/dinov2_vits14_pretrain.pth \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
```

## Configuration

Set `data.dataset_root` in the selected YAML file before training or evaluation.

| Dataset | Configuration |
| --- | --- |
| UHV-4K | `configs/4kbenchmark.yaml` |
| REVIDE | `configs/revide.yaml` |
| HazeWorld | `configs/hazeworld.yaml` |
| Real-world 4K videos | `configs/4krealvideos.yaml` |

## Training

Single-GPU training:

```bash
python train.py --config configs/4kbenchmark.yaml --model ours
```

Multi-GPU training with DistributedDataParallel:

```bash
torchrun --nproc_per_node=8 train.py \
  --config configs/4kbenchmark.yaml \
  --model ours
```

Resume a run with full optimizer and scheduler state:

```bash
python train.py \
  --config configs/4kbenchmark.yaml \
  --model ours \
  --resume checkpoints/4kbenchmark/ours/last.pth
```

## Evaluation

Evaluate paired videos with PSNR, SSIM, LPIPS, runtime, parameter count, and optional FLOPs:

```bash
python test.py \
  --config configs/4kbenchmark.yaml \
  --model ours \
  --checkpoint checkpoints/4kbenchmark/ours/best.pth \
  --save_images
```

Install `thop` to report FLOPs:

```bash
pip install thop
```

Evaluate real-world videos with no-reference image-quality metrics:

```bash
python test_realworld.py \
  --config configs/4krealvideos.yaml \
  --model ours \
  --checkpoint checkpoints/4kbenchmark/ours/best.pth \
  --video_dir /path/to/video_clips \
  --save_images
```

Each directory under `video_clips` is treated as one clip and should contain sequentially numbered frames.

Evaluate temporal consistency:

```bash
python scripts/eval_temporal_consistency.py \
  --config configs/4kbenchmark.yaml \
  --model ours \
  --checkpoint checkpoints/4kbenchmark/ours/best.pth
```

Benchmark memory and throughput:

```bash
python scripts/benchmark_efficiency.py --models Ours
```

## Project Structure

```text
LiBrA-Net/
├── configs/                 # Dataset and experiment configurations
├── data/                    # Video dataset loader
├── models/                  # LiBrA-Net architecture
├── downstream_eval/         # Detection and segmentation evaluation utilities
├── scripts/                 # Efficiency and temporal-consistency benchmarks
├── losses.py                # Training objectives
├── train.py                 # Training entry point
├── test.py                  # Paired evaluation entry point
└── test_realworld.py        # No-reference real-world evaluation
```

## Acknowledgements

- [DINOv2](https://github.com/facebookresearch/dinov2)
- [REVIDE](https://github.com/BookerDeWitt/REVIDE_Dataset)
- [HazeWorld](https://github.com/jiaqixuac/MAP-Net)

