import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.components.data_index import build_brats_file_list
from src.components.transforms import get_train_transforms

data_dir = "data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

files = build_brats_file_list(data_dir)
sample = files[0]

transform = get_train_transforms()
output = transform(sample)

print("Image shape:", output["image"].shape)
print("Label shape:", output["label"].shape)
print("Image dtype:", output["image"].dtype)
print("Label dtype:", output["label"].dtype)
print("Unique labels:", output["label"].unique())