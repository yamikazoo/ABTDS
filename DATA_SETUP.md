# Dataset Setup (BraTS 2020)

## https://www.kaggle.com/code/zeeshanlatif/brain-tumor-segmentation-using-u-net

The BraTS dataset is large (\~8--9 GB), so it is **not included in this
repository**.\
Each team member must download it locally before running the project.

------------------------------------------------------------------------

## 1. Install the Kaggle API

Install the Kaggle Python package:

    pip install kaggle

------------------------------------------------------------------------

## 2. Create a Kaggle API Token (if you need help for this just tell me)

1.  Go to: https://www.kaggle.com/settings/account\
2.  Scroll to the **API** section\
3.  Click **Create New API Token**

This will download a file called:

    kaggle.json

------------------------------------------------------------------------

## 3. Place the API Token in the Kaggle Folder

Create the Kaggle configuration folder and move the token there.

###  PowerShell

    mkdir $HOME\.kaggle
    move $HOME\Downloads\kaggle.json $HOME\.kaggle\

------------------------------------------------------------------------

## 4. Download the Dataset

Run the download script included in the repository:

    python download_data.py

This script downloads the dataset from Kaggle and extracts it
automatically into the `data/` folder.

------------------------------------------------------------------------

## 5. Expected Folder Structure

After downloading, your project should look like this:

    project/
    │
    ├── data/
    │   ├── BraTS2020_TrainingData/
    │   │   ├── BraTS20_Training_001/
    │   │   ├── BraTS20_Training_002/
    │   │   └── ...
    │   │
    │   └── BraTS2020_ValidationData/
    │       ├── BraTS20_Validation_001/
    │       ├── BraTS20_Validation_002/
    │       └── ...

Each patient folder contains **five MRI files**:

    BraTS20_Training_001_flair.nii
    BraTS20_Training_001_t1.nii
    BraTS20_Training_001_t1ce.nii
    BraTS20_Training_001_t2.nii
    BraTS20_Training_001_seg.nii

These correspond to the **four MRI modalities and the segmentation
mask** used by the model.
