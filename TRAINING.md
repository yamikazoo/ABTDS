# Training Guide

## Prerequisites

1. **Install the conda environment:**
   ```bash
   conda env create -f requirements.yml
   conda activate amazing
   ```

2. **Download the dataset (~8-9 GB):**
   ```bash
   export KAGGLE_API_TOKEN=<your-token>
   pip install kaggle
   python download_data.py
   ```
   Get your token from https://www.kaggle.com/settings/account → API → Create New Token.

3. **macOS users — set this in every terminal session (fixes OpenMP conflict):**
   ```bash
   export KMP_DUPLICATE_LIB_OK=TRUE
   ```

## Quick Sanity Check

Run 1 train batch + 1 val batch to verify the pipeline works (~30 seconds):

```bash
python -m src.run --fast_dev_run --config config/default.yaml
```

You should see a table showing SegResNet with 1.2M params, followed by 1 training step and 1 validation step completing without errors.

## Full Training Run

```bash
python -m src.run --config config/default.yaml
```

Default settings: 50 epochs, batch_size 1, lr 0.0001, 4 dataloader workers.

### Override settings via CLI

```bash
python -m src.run --config config/default.yaml --epochs 100 --lr 0.0005
```

CLI arguments always override values from the YAML config.

### Enable mixed precision (faster on supported GPUs)

```bash
python -m src.run --config config/default.yaml --amp
```

## Monitoring with TensorBoard

In a second terminal:

```bash
conda activate amazing
tensorboard --logdir outputs/logs/brats_baseline
```

Open http://localhost:6006 in your browser. Metrics appear after the first epoch completes.

### Metrics to watch

| Metric | What it means | Healthy range |
|---|---|---|
| `train_loss` | Training loss (DiceCE). Should decrease over time. | Starts ~2.0, drops toward 0.3–0.6 |
| `val_loss` | Validation loss. Should track train_loss downward. | Similar to train_loss |
| `val_mean_dice` | Dice overlap score (0 = no overlap, 1 = perfect). Averaged across 3 tumor sub-regions. | > 0.5 is decent, > 0.7 is good |

**Signs of overfitting:** `train_loss` keeps dropping but `val_loss` stops decreasing or starts rising.

## Monitoring with Weights & Biases

Add the `--wandb` flag to enable W&B logging:

```bash
python -m src.run --config config/default.yaml --wandb --wandb_project brats-segmentation
```

You'll be prompted to log in on first use. Results are viewable at https://wandb.ai.

Optional: name your run with `--wandb_run_name "my-experiment"`.

## Hyperparameter Sweeps

Run an automated hyperparameter search using W&B Sweeps:

```bash
python sweep.py --count 5
```

This runs 5 trials with different learning rates (Bayesian search), optimizing for best `val_mean_dice`. The sweep config is in `config/sweep.yaml`.

To run more trials on an existing sweep:

```bash
python sweep.py --sweep_id <id-from-first-run> --count 10
```

## Checkpoints

The top 3 checkpoints (by best `val_mean_dice`) are saved to `outputs/checkpoints/`. Filenames include the epoch and dice score, e.g.:

```
brats-epoch=12-val_mean_dice=0.6234.ckpt
```

## Expected Training Time

| Hardware | Approx. time per epoch | 50 epochs |
|---|---|---|
| Apple M-series (MPS) | ~10 min | ~8 hours |
| NVIDIA GPU (e.g. RTX 3090) | ~2-3 min | ~2-3 hours |
| Google Colab (T4) | ~5 min | ~4 hours |

Times vary based on the specific hardware and number of dataloader workers.
