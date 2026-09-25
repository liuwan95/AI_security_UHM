# Fooling automated surveillance cameras: adversarial patches to attack person detection

This folder contains my ECE 491E reproduction/demo based on the paper **“Fooling automated surveillance cameras: adversarial patches to attack person detection”** by Simen Thys, Wiebe Van Ranst, and Toon Goedemé (2019).

Paper: https://arxiv.org/abs/1904.08653  

The goal of the demo is to train a universal adversarial patch against the **person** class of YOLOv2, evaluate whether the patch lowers person-detection confidence, and demonstrate the patch with a live camera. This is an educational reproduction rather than an exact reimplementation of every experiment in the paper.

## Important: files not included in the repository

The repository intentionally does **not** include the following large files:

- YOLOv2 pretrained weights (`weights/yolo.weights`)
- INRIA Person dataset images (`data/`)

They must be downloaded after cloning. The setup steps below create the expected directory structure automatically.

## Repository structure

```text
Fooling automated surveillance cameras: adversarial patches to attack person detection/
├── cfg/
│   └── yolo.cfg
├── non_printability/
│   └── 30values.txt
├── patches/
│   ├── universal_patch_cls.png
│   ├── universal_patch_obj.png
│   └── universal_patch_obj_cls.png
├── src/
│   ├── cfg.py
│   ├── darknet.py
│   ├── evaluate_universal_patch.py
│   ├── live_patch_demo.py
│   ├── patch_utils.py
│   ├── region_loss.py
│   ├── train_universal_patch.py
│   ├── utils.py
│   └── yolo_loss.py
├── download_inria_subset.py
├── requirements.txt
└── README.md
```

The three PNG files in `patches/` are previously trained examples. Training again will overwrite the patch corresponding to the selected attack type.

## 1. Clone and enter the repository

Clone the class repository and enter this paper's folder. For example:

```bash
git clone https://github.com/liuwan95/AI_security_UHM.git
cd AI_security_UHM
cd "Fooling automated surveillance cameras: adversarial patches to attack person detection"
```

If the folder is placed somewhere else inside the class repository, change the final `cd` command accordingly.

## 2. Create a Python virtual environment

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The code runs on CPU, but training is much faster with an NVIDIA GPU. If the default PyTorch installation does not detect your GPU, install the PyTorch build that matches your CUDA setup from https://pytorch.org/get-started/locally/ and then install the remaining requirements.

You can check CUDA detection with:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 3. Download the YOLOv2 weights

Create the `weights` directory and download the standard YOLOv2 weights. The expected final path is:

```text
weights/yolo.weights
```

### Linux/macOS

```bash
mkdir -p weights
wget -O weights/yolo.weights https://pjreddie.com/media/files/yolo.weights
```

If `wget` is unavailable, use:

```bash
curl -L https://pjreddie.com/media/files/yolo.weights -o weights/yolo.weights
```

### Windows PowerShell

```powershell
New-Item -ItemType Directory -Force weights
Invoke-WebRequest -Uri "https://pjreddie.com/media/files/yolo.weights" -OutFile "weights/yolo.weights"
```

After downloading, verify that `weights/yolo.weights` exists before running the training or evaluation scripts.

## 4. Download the INRIA Person images

The included downloader creates a reproducible subset of positive INRIA Person training images using a public GitHub mirror. It downloads:

- **100 training images** into `data/train_images/`
- **40 validation images** into `data/validation_images/`

Run:

```bash
python download_inria_subset.py
```

Expected directory structure afterward:

```text
data/
├── train_images/
│   └── ... 100 images ...
└── validation_images/
    └── ... 40 images ...
```

The split uses a fixed random seed (`42`), so running the downloader again should create the same train/validation split as long as the source image list is unchanged.

## 5. Train a universal adversarial patch

The main training script is:

```bash
python src/train_universal_patch.py
```

At the top of `src/train_universal_patch.py`, `ATTACK_TYPE` selects which YOLO component is attacked:

```python
ATTACK_TYPE = "cls"
```

Supported values are:

- `"obj"` - reduce YOLO objectness
- `"cls"` - reduce person-class confidence
- `"obj_cls"` - combine the objectness and classification objectives

The default configuration uses the `cls` attack and saves the result as:

```text
patches/universal_patch_cls.png
```

The training code also includes the two regularization terms used by the paper's physical-patch formulation:

- **NPS (Non-Printability Score):** encourages colors that are easier for a printer to reproduce.
- **TV (Total Variation):** encourages spatial smoothness instead of extreme pixel-to-pixel changes.

Random rotation, scale, brightness, contrast, and noise are also applied during training to make the patch less dependent on one exact image condition.

Important settings such as `PATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`, `NPS_WEIGHT`, and `TV_WEIGHT` are near the top of `src/train_universal_patch.py`.

## 6. Evaluate the trained patch

Run:

```bash
python src/evaluate_universal_patch.py
```

By default the evaluator loads:

```text
patches/universal_patch_cls.png
```

and evaluates up to 40 images from:

```text
data/validation_images/
```

It compares three conditions:

1. **Clean** - original image with no patch
2. **Random** - image with a random patch of the same approximate size
3. **Universal** - image with the trained adversarial patch

The script reports:

- **Average person confidence:** the mean confidence of the strongest person detection across the evaluated images.
- **Detection rate:** the percentage of images whose strongest person detection remains at or above the configured confidence threshold (`0.40`).
- **Average confidence drop:** clean confidence minus patched confidence. A larger drop means the patch reduced the detector's confidence more strongly.

To evaluate another saved attack variant, change `UNIVERSAL_PATCH_PATH` near the top of `src/evaluate_universal_patch.py`.

## 7. Run the live camera demo

Connect a webcam and run:

```bash
python src/live_patch_demo.py
```

The live demo uses:

```text
patches/universal_patch_cls.png
```

and camera index `0` by default. If the wrong camera opens, change:

```python
CAMERA_INDEX = 0
```

near the top of the script.

The demo shows how YOLOv2 responds to a person in real time while the trained patch is present. GPU acceleration is strongly recommended for a smoother frame rate.

## Reference

Simen Thys, Wiebe Van Ranst, and Toon Goedemé, **“Fooling automated surveillance cameras: adversarial patches to attack person detection,”** arXiv:1904.08653, 2019.

This code is provided for coursework, research, and educational demonstration of adversarial machine-learning behavior.
