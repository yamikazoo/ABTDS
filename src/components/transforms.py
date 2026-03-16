'''
2. Tensor preprocessing

Currently, data_index.py only returns file paths:

{
    "image": [
        "..._flair.nii",
        "..._t1.nii",
        "..._t1ce.nii",
        "..._t2.nii",
    ],
    "label": "..._seg.nii"
}

These are just paths, not usable by the model yet.

We must load the MRI files and convert them into tensors with the correct shape:

image: [4, H, W, D]      (4 MRI modalities as channels)
label: [H, W, D] or [1, H, W, D]

This preprocessing is handled using MONAI transforms.
'''

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    CropForegroundd,
    NormalizeIntensityd,
    RandFlipd,
    RandRotate90d,
    RandSpatialCropd,
    MapLabelValued, 
)

def get_train_transforms(patch_size=(96, 96, 96)):
    return Compose([
        #Loads the MRI files and segmentation mask from the paths
        LoadImaged(keys=["image", "label"]),
        
        #Makes sure channels come first, so the 4 MRI modalities become channel-first input
        EnsureChannelFirstd(keys=["image", "label"]),
        
        #Re-maps label 4 (Enhancing Tumor) to 3 to prevent index out of bounds error
        MapLabelValued(keys="label", orig_labels=[4], target_labels=[3]),
        
        #Standardizes MRI intensities so scans from different patients are more comparable
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        
        #Removes large empty background regions
        CropForegroundd(keys=["image", "label"], source_key="image"),
        
        #Extracts a smaller 3D patch like 96x96x96 for training
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=patch_size,
            random_size=False,
        ),
       
       #Data augmentation
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        EnsureTyped(keys=["image", "label"]),
    ])


#Preparing data for validation
def get_val_transforms():
    return Compose([
        #Loads the MRI files (4 modalities) and the segmentation mask from disk into memory.
        LoadImaged(keys=["image", "label"]),
        
        #Reorders the data so channels come first. The 4 MRI modalities become a 4-channel image tensor.
        EnsureChannelFirstd(keys=["image", "label"]),

        #Re-maps label 4 (Enhancing Tumor) to 3 to prevent index out of bounds error
        MapLabelValued(keys="label", orig_labels=[4], target_labels=[3]),

        #Normalizes MRI intensity values so scans from different patients are on a similar scale.
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        
        #Removes empty background areas around the brain to reduce unnecessary space.
        CropForegroundd(keys=["image", "label"], source_key="image"),
        
        #Converts the data into PyTorch tensors so it can be used by the neural network.
        EnsureTyped(keys=["image", "label"]),
    ])