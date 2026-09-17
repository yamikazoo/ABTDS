# ABTDS (Automated Brain Tumor Detection & Segmentation) - SFU CMPT 419 Term Project 

This project develops a deep learning pipeline using MONAI's SegResNet to automatically segment brain tumor sub-regions from multi-modal MRI scans. Using the BraTS 2020 dataset and a hybrid Dice-Cross Entropy loss function, the model produces volumetric segmentation masks across 4 classes (Background, NCR, ED, ET) which can be used to assist clinicians in treatment planning.

## Contributors

- Calvin Weng
- Alex Jiang 
- Rodrigo Anasco 
- The Vi Phung
- Tallal Mohar

## Important Links

 [Project report](https://www.overleaf.com/project/69c4577714adfcde19a4cf59)

## Video/demo
[Link to YT video](https://youtu.be/YTRF01697X0)

## Table of Contents
1. [Project Structure](#project_structure)

2. [Installation and Environment Setup](#installation)

3. [Dataset Setup](#dataset_setup)

4. [Training](#training)

4. [Tests](#tests)

6. [Full Run Guide](#fr_guide)

<a name="project_structure"></a>
## 1. Project Structure

```
repository
├── src/
│   ├── run.py                      ## Training entry point
│   ├── train.py                    ## BraTSModel (SegResNet + DiceCELoss)
│   ├── utils/
│   │   └── post_process.py         ## Connected-component cleaner for tiny prediction specks
│   └── components/
│       ├── data_index.py           ## Scans BraTS folders into file lists
│       ├── data_module.py          ## Lightning DataModule (train/val split, loaders)
│       ├── transforms.py           ## MONAI transform pipelines
│       └── metrics.py              ## Dice and HD95 for WT/TC/ET regions
├── config/
│   ├── default.yaml                ## Default training hyperparameters
│   └── sweep.yaml                  ## W&B hyperparameter sweep config
├── sweep.py                        ## W&B sweep launcher script
├── tests/
│   ├── test_transforms.py          ## Validates transform pipeline shapes/dtypes
│   └── test_data_module.py         ## Validates dataloader and split
├── outputs/                        ## (git-ignored) checkpoints and logs
├── sample_data/                    ## 1 sample patient for quick testing
├── data/                           ## (git-ignored) BraTS 2020 dataset
├── requirements.yml                ## Conda environment
├── requirements.txt                ## Pip requirements
├── download_data.py                ## Dataset download script
├── TRAINING.md                     ## Full training guide
└── DATA_SETUP.md                   ## Dataset setup instructions
```

<a name="installation"></a>
## 2. Installation and Environment Setup

```bash
git clone <repo-url>
cd 2026_1_project_20
conda env create -f requirements.yml
conda activate amazing
```

<a name="dataset_setup"></a>
## 3. Dataset Setup

Download the BraTS 2020 dataset (~8-9 GB):

```bash
export KAGGLE_API_TOKEN=<your-token>
pip install kaggle
python download_data.py
```

Get your Kaggle token from https://www.kaggle.com/settings/account → API → Create New Token.

See [DATA_SETUP.md](DATA_SETUP.md) for detailed instructions.

### Quick Test with Sample Data (no download needed)

A single sample patient is included in `sample_data/` for quick verification:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m src.run --fast_dev_run --data_dir sample_data/MICCAI_BraTS2020_TrainingData
```

<a name="training"></a>
## 4. Training

**Sanity check** (1 batch, ~30 seconds):
```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m src.run --fast_dev_run --config config/default.yaml
```

**Full training run:**
```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m src.run --config config/default.yaml
```

**With W&B logging:**
```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m src.run --config config/default.yaml --wandb
```

**Monitor with TensorBoard** (in a second terminal):
```bash
tensorboard --logdir outputs/logs/brats_baseline
```

See [TRAINING.md](TRAINING.md) for the full training guide including metrics, sweeps, and expected training times.

<a name="tests"></a>
## 5. Tests

Requires the dataset to be downloaded:

```bash
python tests/test_transforms.py
python tests/test_data_module.py
```

<a name="fr_guide"></a>
## 6. Full Run Guide

This section is intended for grading/demo. If you follow the commands in order, the project can be run end-to-end.

1) **Clone and create environment**

```bash
git clone <repo-url>
cd 2026_1_project_20
conda env create -f requirements.yml
conda activate amazing
```

2) **Download BraTS 2020 data**

```bash
export KAGGLE_API_TOKEN=<your-token>
python download_data.py
```

Expected data path after download:
`data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData`

3) **Run a sanity check (recommended first)**

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m src.run --fast_dev_run --config config/default.yaml
```

4) **Run training**

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m src.run --config config/default.yaml
```

Outputs are saved to:
- `outputs/checkpoints/` (best checkpoints by validation Dice)
- `outputs/logs/brats_baseline/` (TensorBoard logs)

5) **Run tests**

```bash
python tests/test_transforms.py
python tests/test_data_module.py
```

6) **Optional: clean prediction masks using connected components**

If you have predicted segmentation masks in NIfTI format (`.nii`/`.nii.gz`), remove tiny isolated regions:

```bash
python -m src.utils.post_process --input_dir <pred_dir> --output_dir <clean_dir> --min_size 50
```

Useful flags:
- `--classes 1 2 3` to clean only specific class IDs
- `--connectivity 1` for stricter neighborhood connectivity
