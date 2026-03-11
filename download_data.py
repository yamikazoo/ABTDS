import kaggle

#This is to download the dataset
kaggle.api.dataset_download_files(
    "awsaf49/brats20-dataset-training-validation",
    path="data",
    unzip=True
)

print("Download complete.")