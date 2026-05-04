# LiBrA-Net (If the online folder view appears empty, please download the ZIP file — all contents are intact. This is a known Anonymous GitHub display issue.)

**LiBrA-Net: Lie-Algebraic Bilateral Affine Fields for Real-Time 4K Video Dehazing**

A real-time video dehazing framework that factorizes the spatiotemporal affine field into a pair of compact bilateral sub-grids, fuses them in the Lie algebra $\mathfrak{gl}(3)$ via a Cayley parameterization, and recovers high-frequency detail through a lightweight input-guided branch. LiBrA-Net achieves state-of-the-art dehazing quality at native 4K resolution (**25 FPS**, 6.12M parameters, 282 GFLOPs on a single GPU).

## UHV-4K Dataset

We release **UHV-4K**, the first paired 4K video dehazing benchmark with depth, transmission, and optical-flow annotations. It contains 100 native 3840×2160 videos across seven scene categories.

Details on synthesis pipeline and quality checks are described in the paper.

## Installation

### Requirements

- Python >= 3.10
- PyTorch >= 2.1
- CUDA-compatible GPU (recommended)

### Setup

```bash
git clone https://github.com/ANO_REPO/LiBrA-Net.git
cd LiBrA-Net
pip install -r requirements.txt
```

### Pre-trained Weights

Download the DINOv2 ViT-S/14 pre-trained weights (required for perceptual loss during training):

```bash
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
```

## Dataset Preparation

The expected dataset structure follows the format:

```
dataset_root/
├── Train/  (or train/)
│   ├── hazy/
│   │   ├── sequence_001/
│   │   │   ├── 00000.jpg
│   │   │   └── ...
│   │   └── sequence_002/
│   └── gt/
│       ├── sequence_001/
│       └── sequence_002/
└── Test/  (or test/)
    ├── hazy/
    └── gt/
```

### Supported Datasets

| Dataset | Config | Description |
|---------|--------|-------------|
| [REVIDE](https://github.com/BookerDeWitt/REVIDE_Dataset) | `configs/revide.yaml` | Real-world video dehazing dataset |
| [HazeWorld](https://github.com/jiaqixuac/MAP-Net) | `configs/hazeworld.yaml` | Large-scale haze synthesis dataset |
| UHV-4K | `configs/4kbenchmark.yaml` | Our 4K video dehazing benchmark |

Update `data.dataset_root` in the corresponding config YAML to point to your dataset path.

## Training

### Single GPU

```bash
python train.py --config configs/revide.yaml --model ours
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=8 train.py --config configs/revide.yaml --model ours
```

## Evaluation

### Standard Evaluation (with GT)

```bash
python test.py \
    --config configs/revide.yaml \
    --model ours \
    --checkpoint checkpoints/revide/ours/last.pth \
    --save_images
```

Metrics computed: **PSNR**, **SSIM**, **LPIPS**, **FPS**, **GFLOPs**.

### Real-World Evaluation (no GT)

```bash
python test_realworld.py \
    --config configs/4krealvideos.yaml \
    --model ours \
    --checkpoint checkpoints/4kbenchmark/ours/last.pth \
    --video_dir ./path/to/real_videos \
    --save_images
```

Metrics computed: **MUSIQ**, **CLIP-IQA+**, **BRISQUE**, **NIQE**.


## Acknowledgements

- [DINOv2](https://github.com/facebookresearch/dinov2) for the perceptual loss backbone
- [REVIDE](https://github.com/BookerDeWitt/REVIDE_Dataset) dataset
- [HazeWorld](https://github.com/jiaqixuac/MAP-Net) dataset
- Comparison method authors for their original implementations
