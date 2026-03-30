# Dataset Setup (BraTS 2020)

The BraTS dataset is large (~8-9 GB), so it is **not included in this repository**.
Each team member must download it locally before running the project.

---

## 1. Install the Kaggle API

```bash
conda activate amazing
pip install kaggle
```

---

## 2. Create a Kaggle API Token

1. Go to: https://www.kaggle.com/settings/account
2. Scroll to the **API** section
3. Click **Create New API Token**
4. Kaggle will display your token on screen — copy it

---

## 3. Set the API Token

Set the token as an environment variable in your terminal:

### macOS / Linux

```bash
export KAGGLE_API_TOKEN=<your-token>
```

### Windows (PowerShell)

```powershell
$env:KAGGLE_API_TOKEN = "<your-token>"
```

---

## 4. Download the Dataset

Run the download script included in the repository:

```bash
python download_data.py
```

This downloads the dataset from Kaggle and extracts it automatically into the `data/` folder.

---

## 5. Expected Folder Structure

After downloading, your project should look like this:

```
project/
│
├── data/
│   ├── BraTS2020_TrainingData/
│   │   └── MICCAI_BraTS2020_TrainingData/
│   │       ├── BraTS20_Training_001/
│   │       ├── BraTS20_Training_002/
│   │       └── ...
│   │
│   └── BraTS2020_ValidationData/
│       ├── BraTS20_Validation_001/
│       ├── BraTS20_Validation_002/
│       └── ...
```

Each patient folder contains **five MRI files**:

```
BraTS20_Training_001_flair.nii
BraTS20_Training_001_t1.nii
BraTS20_Training_001_t1ce.nii
BraTS20_Training_001_t2.nii
BraTS20_Training_001_seg.nii
```

These correspond to the **four MRI modalities and the segmentation mask** used by the model.

**Note:** Patient 355 has a non-standard segmentation filename and is automatically skipped by the data pipeline.
