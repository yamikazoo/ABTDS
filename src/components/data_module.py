"""
DataModule

This file connects the BraTS dataset to the model.

What it does:
1. Reads all patient file paths from data_index.py
2. Splits the dataset into training and validation sets
3. Applies the correct MONAI transforms
4. Creates DataLoaders that give batches to the model

Why we need it:
Instead of manually loading and splitting the data every time,
the DataModule keeps everything organized in one place.
This makes it easy for the rest of the team to train and validate the model.
"""

import random
from monai.data import Dataset
from torch.utils.data import DataLoader

from src.components.data_index import build_brats_file_list
from src.components.transforms import get_train_transforms, get_val_transforms


class BraTSDataModule:
    """
    Manages the BraTS dataset pipeline.

    Main responsibilities:
    - build the list of patient files
    - split patients into training and validation sets
    - apply transforms
    - create dataloaders
    """

    def __init__(
        self,
        data_dir,
        train_split=0.8,
        batch_size=1,
        num_workers=0,
        patch_size=(96, 96, 96),
        seed=42,
    ):
        """
        Stores all the configuration needed for the data pipeline.

        Parameters:
        - data_dir: folder containing the BraTS patient folders
        - train_split: percentage of data used for training
        - batch_size: number of samples per training batch
        - num_workers: number of worker processes for loading data
        - patch_size: 3D patch size for training transforms
        - seed: random seed for reproducible splitting
        """
        self.data_dir = data_dir
        self.train_split = train_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.seed = seed

        # These will be filled later in setup()
        self.train_files = None
        self.val_files = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self):
        """
        Prepares the datasets.

        What happens here:
        1. Build the full patient file list
        2. Shuffle the patients
        3. Split them into train and validation sets
        4. Create MONAI datasets with the correct transforms
        """
        # Build the full list of BraTS patient files
        files = build_brats_file_list(self.data_dir)

        # Shuffle the list so training/validation split is random
        random.seed(self.seed)
        random.shuffle(files)

        # Compute the split index
        split_idx = int(len(files) * self.train_split)

        # Split into training and validation file lists
        self.train_files = files[:split_idx]
        self.val_files = files[split_idx:]

        # Create the MONAI training dataset
        self.train_dataset = Dataset(
            data=self.train_files,
            transform=get_train_transforms(self.patch_size)
        )

        # Create the MONAI validation dataset
        self.val_dataset = Dataset(
            data=self.val_files,
            transform=get_val_transforms()
        )

    def train_dataloader(self):
        """
        Returns the training dataloader.

        This dataloader:
        - uses training transforms
        - shuffles batches
        - returns small 3D patches for model training
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """
        Returns the validation dataloader.

        This dataloader:
        - uses validation transforms
        - does not shuffle
        - returns full cropped validation volumes
        """
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    