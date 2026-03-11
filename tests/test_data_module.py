import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.components.data_module import BraTSDataModule


data_dir = "data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

dm = BraTSDataModule(
    data_dir=data_dir,
    batch_size=1,
    num_workers=0,
    patch_size=(96, 96, 96),
)

dm.setup()

train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()

train_batch = next(iter(train_loader))
val_batch = next(iter(val_loader))

print("Train image shape:", train_batch["image"].shape)
print("Train label shape:", train_batch["label"].shape)

print("Val image shape:", val_batch["image"].shape)
print("Val label shape:", val_batch["label"].shape)

print("Number of training patients:", len(dm.train_files))
print("Number of validation patients:", len(dm.val_files))