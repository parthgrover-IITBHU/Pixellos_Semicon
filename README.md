# Image Restoration / Super-Resolution


## Model

The final notebook architecture uses:

- Single-channel input.
- Per-image 1st/99th percentile clipping and normalization.
- 32-channel feature extraction.
- Encoder Restormer stages: `(4, 6, 6)` blocks.
- Bottleneck: `8` Restormer blocks at `8C` channels.
- Decoder Restormer stages: `(6, 6, 4)` blocks.
- Decoder upsampling uses `3x3 convolution -> PixelShuffle(2)`.
- A final PixelShuffle(2) and residual connection to bicubic upsampling.
- Output is transformed back using the saved percentile range and clipped to `[0,1]`.

The active training loss is:

`Charbonnier pixel loss + Sobel-gradient L1 edge loss`

The optimizer is AdamW with `lr=1e-4`, betas `(0.9, 0.999)`,
weight decay `1e-3`, gradient clipping at `1.0`, and cosine annealing
to `1e-7`. Training uses a 90/10 train/validation split with seed `42`.
The best checkpoint is selected using validation PSNR.

## Repository layout

```text
.
├── README.md
├── model.py
├── train.py
├── run.py
├── requirements.txt
├── model_weights/best_model.pt                 
├── restored_test_outputs/         
```

## Environment

Python 3.10+ is recommended.

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

For PyTorch, use the build appropriate for the reviewer's CUDA/CPU environment.

## Data format

The model loads images from directories containing `.npy` files.

Training expects:

```text
data/
├── lr-train/
│   ├── image_001.npy
│   ├── image_002.npy
│   └── ...
└── gt-train/
    ├── image_001.npy
    ├── image_002.npy
    └── ...
```

The `.npy` images are expected to be single-channel arrays with shape
`(1,H,W)` (the model's input is converted to `(B,1,H,W)`).

Test data is a directory of `.npy` files:

```text
data/
└── noisy-test/
    ├── image_001.npy
    ├── image_002.npy
    └── ...
```

## Inference

Place the trained `best_model.pt` next to `run.py`.

Run :

```bash
python run.py path/to/noisy-test path/to/restored_outputs --weights best_model.pt
```


The script:

1. Loads every `.npy` file in the supplied test directory.
2. Loads the trained `state_dict`.
3. Runs the restoration model.
4. Writes each restored image as a grayscale .npy into the requested output directory.

No source-code edits are required.

## Reproducing training

Training can be reproduced with:

```bash
python train.py \
    --lr-dir path/to/lr-train \
    --gt-dir path/to/gt-train \
    --output best_model.pt \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-4 \
    --channels 32
```

The script performs the same seeded 80/20 train/validation split used in the original
notebook and saves the checkpoint with the best validation PSNR.
