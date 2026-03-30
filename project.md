# BraTS Brain Tumor Segmentation Project

## Project Overview

**Automated Brain Tumor Detection & Segmentation** — A deep learning project that automatically segments brain tumor sub-regions (edema, core, and enhancing tumor) from multi-modal MRI scans using the BraTS 2020 dataset and MONAI for accelerated 3D processing.

## Motivation

### Clinical Motivation

Manual segmentation of 3D MRI scans is highly time-consuming for radiologists and prone to human error. Automating this process directly improves surgical planning and patient prognosis.

### Technical Feasibility

- The BraTS 2020 dataset provides high-quality, radiologist-vetted annotations
- High-level medical imaging frameworks (MONAI) bypass low-level data engineering roadblocks
- Focus can remain on model training, loss optimization, and evaluation

## Dataset Information

### Source

BraTS 2020 Training Data (Kaggle / UPenn)  
https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation

### Volume & Scope

- **369 training subjects** total
- **Initial prototype: 50-patient subset** to ensure pipeline correctness before scaling

### Inputs (Features)

Four 3D MRI modalities per subject:

- T1
- T1-contrast enhanced (T1CE)
- T2
- FLAIR

### Outputs (Targets)

One radiologist-vetted ground truth segmentation mask per subject with 4 classes:

- **0**: Background
- **1**: Necrotic / Non-enhancing Tumor Core (NCR)
- **2**: Peritumoral Edema (ED)
- **4**: GD-enhancing Tumor (ET)

_Note: Label 3 is intentionally absent in the BraTS standard; this gap is remapped during preprocessing._

---

## Methods & Techniques

### 1. Data Preprocessing & Augmentation (via MONAI)

Medical MRI data requires specific handling using MONAI's pre-built transform pipelines.

**Modality Fusion & Normalization:**

- Concatenate four MRI modalities into a 4-channel input volume
- Apply `NormalizeIntensityd` (Z-score normalization) to handle varying MRI scanner signals

**Cropping:**

- Use `CropForegroundd` to automatically strip empty background space
- Significantly reduces computational overhead

**Data Augmentation:**

- `RandRotated`, `RandFlipd`, and `RandSpatialCropd` extract smaller 3D blocks
- Target patch size: **96 × 96 × 96 voxels** to fit within standard GPU memory

### 2. Model Architecture: SegResNet

**Philosophy:** Use off-the-shelf implementations to guarantee mathematical correctness from Day 1.

**Implementation:** MONAI's `SegResNet` or `UNet`  
**Structure:**

- **Encoder Path:** Captures global context
- **Decoder Path:** Produces voxel-level predictions
- **Skip Connections:** Maintain precise tumor boundaries

### 3. Loss Function Strategy

Standard Cross-Entropy fails in medical imaging due to severe class imbalance (tumors occupy a tiny fraction of the brain volume).

**Solution:** MONAI's `DiceCELoss`

- Combines **Dice Loss** (optimizes for ground-truth volume overlap)
- Combined with **Cross-Entropy** (provides smooth gradients during early training)

### 4. Training & Hardware Environment

**Framework:**

- PyTorch + PyTorch Lightning
- Lightning eliminates boilerplate loops and prevents common bugs

**Optimization:**

- Automatic Mixed Precision (AMP) dynamically switches between half and single precision
- Reduces VRAM consumption and accelerates training

**Inference:**

- MONAI's `sliding_window_inference` stitches 96×96×96 patches back into full brain predictions during testing

### 5. Evaluation Metrics

- **Dice Similarity Coefficient (DSC):** Volume accuracy
- **95th Percentile Hausdorff Distance (HD95):** Boundary precision

---

## Team Division of Labor (1-Month Sprint)

To enable 5 students to work concurrently without merge conflicts:

| Role                                       | Responsibilities                                                                                                                |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| **Rodrigo (Data Engineer)**                | Downloads BraTS data, writes MONAI transform pipelines (loading, cropping, normalizing, augmenting), builds PyTorch DataLoaders |
| **Yami (Model & Training Lead)**           | Sets up PyTorch Lightning module, initializes MONAI SegResNet, configures DiceCELoss, implements AMP optimizer                  |
| **Vi (Validation & Metrics Specialist)**   | Writes validation loop, sets up sliding_window_inference, implements DSC and HD95 calculation                                   |
| **Tallal (MLOps & Cloud)**                 | Sets up GPU environment (Colab Pro, AWS, local lab servers), integrates W&B/TensorBoard logging, handles hyperparameter tuning  |
| **Alex (Post-Processing & Documentation)** | Implements connected component analysis, creates visualization tools, writes final documentation as Project Manager             |

---

## Project Structure

```
braTS-tumor-segmentation/
├── data/                   # (Git-ignored) Raw and processed .nii.gz files
├── notebooks/              # For initial EDA and testing MONAI transforms
│   └── 01_eda_and_visualization.ipynb
├── src/                    # Main source code
│   ├── __init__.py
│   ├── components/         # Modular building blocks
│   │   ├── transforms.py   # MONAI preprocessing/augmentation pipelines
│   │   ├── data_module.py  # PyTorch Lightning DataModule (DataLoaders)
│   │   ├── model.py        # SegResNet definition
│   │   └── metrics.py      # Dice and HD95 calculation logic
│   ├── utils/              # Helper functions
│   │   ├── visualization.py # Overlay masks on MRI slices
│   │   └── post_process.py # Connected Component Analysis
│   └── train.py            # Main entry point
├── tests/                  # Sanity checks (e.g., tensor shape verification)
├── configs/                # YAML files for hyperparameters
├── .gitignore              # Ignore .nii.gz, .pyc, outputs/
├── requirements.yml        # Dependencies (monai, torch, pytorch-lightning)
└── README.md               # "How to Run" instructions
```

---

## Module Responsibilities

### Data & Preprocessing (Rodrigo)

- `data/`: Ensure .nii.gz files are structured correctly (imagesTr, labelsTr)
- `src/components/transforms.py`: MONAI dictionary-based transforms
- `src/components/data_module.py`: Wrap into LightningDataModule
- **Critical Interface:** Define exact key names (e.g., `"image"`, `"label"`)

### Modeling & Training (Yami & Tallal)

- `src/components/model.py`: LightningModule with SegResNet initialization
- `src/train.py`: Main execution logic + W&B logging integration
- `configs/`: Hyperparameter YAML files managed by Tallal

### Validation & Evaluation (Vi)

- `src/components/metrics.py`: Dice and HD95 calculation logic
- `src/train.py` (validation_step): Ensure sliding_window_inference calculates metrics on full brain volumes, not patches

### Post-Processing & Reporting (Alex)

- `src/utils/post_process.py`: Remove false-positive tumor "islands" via connected component analysis
- `src/utils/visualization.py`: Overlay tumor masks on 2D MRI slices
- `README.md`: Clear instructions for running the entire pipeline

---

## Safe Zones Strategy

Prevent merge conflicts and blocking:

1. **Rodrigo finishes first:** Other team members cannot start training until `data_module` is functional
2. **Tallal is the Gatekeeper:** All new dependencies must be approved by Tallal before installation (maintains `requirements.yml`)
3. **Alex works at the End:** Can start writing visualization code using ground-truth labels while waiting for model completion

---

## Key Technologies

- **Deep Learning:** PyTorch, PyTorch Lightning
- **Medical Imaging:** MONAI (Medical Open Network for AI)
- **Data Handling:** NIfTI format (.nii.gz)
- **Experiment Tracking:** Weights & Biases (W&B) or TensorBoard
- **Hardware:** GPU-accelerated training (CUDA/AMP)

---

## Completed Work

- ✅ Data download script and BraTS file indexing (Rodrigo)
- ✅ MONAI transform pipelines — train and val (Rodrigo)
- ✅ Lightning DataModule with 80/20 split (Rodrigo)
- ✅ SegResNet model + DiceCELoss training loop (Yami)
- ✅ Sliding window validation + Dice metric logging (Vi)
- ✅ BraTSMetrics class for WT/TC/ET with Dice and HD95 (Vi)
- ✅ Training entry point with CLI args (src/run.py)
- ✅ W&B integration with --wandb flag (Tallal)
- ✅ YAML-based hyperparameter config system (Tallal)
- ✅ W&B Sweep config + launcher script for HP tuning (Tallal)
- ✅ Training documentation (TRAINING.md) (Tallal)
- ✅ Pipeline verified end-to-end on full 370-patient dataset
- ✅ Basic test scripts for transforms and data module

## Remaining Work

### Alex — Post-Processing & Visualization
- [ ] `src/utils/post_process.py` — Connected component analysis to remove small false-positive tumor islands from predictions
- [ ] `src/utils/visualization.py` — Overlay predicted tumor masks on 2D MRI slices for qualitative evaluation
- [ ] Final documentation and reproducibility instructions in the project report

### Yami — Model & Training Improvements
- [ ] Learning rate scheduler (e.g., CosineAnnealingLR or WarmRestarts) in `configure_optimizers()` — current constant lr of 1e-4 limits convergence
- [ ] Evaluate whether extracting the model into a separate `src/components/model.py` improves clarity (currently lives in `src/train.py`)

### Vi — Validation & Metrics
- [ ] Log per-region Dice scores (WT, TC, ET) separately during validation, not just the averaged `val_mean_dice`
- [ ] Integrate BraTSMetrics (HD95) into the validation loop — currently only Dice is computed during training

### Team — Final Steps
- [ ] Full training run on a machine with a dedicated GPU (NVIDIA recommended) to achieve target Dice > 0.7
- [ ] Hyperparameter sweep via `python sweep.py --count 5` to find optimal lr
- [ ] Record demo video (1:40–2 min) showcasing predictions
- [ ] Complete and submit the Overleaf project report

## Success Criteria

- ✓ Pipeline runs end-to-end on full dataset without errors
- ✓ Model achieves baseline Dice scores for each tumor region
- ✓ Validation uses proper sliding-window inference on full volumes
- ○ Clear visualization of predictions for final report
- ✓ Complete documentation for reproducibility
