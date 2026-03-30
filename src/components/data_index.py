'''
1. Data Indexing
This is for convert the folder structure into a dataset list, this would be the expected format:

[
  {
    "image": [
      "data/.../BraTS20_Training_001_flair.nii",
      "data/.../BraTS20_Training_001_t1.nii",
      "data/.../BraTS20_Training_001_t1ce.nii",
      "data/.../BraTS20_Training_001_t2.nii"
    ],
    "label":
      "data/.../BraTS20_Training_001_seg.nii"
  },

  {
    "image": [...],
    "label": ...
  }
]

MONAI reads this and loads the images

'''
from pathlib import Path


def build_brats_file_list(data_dir):
    data_dir = Path(data_dir)

    patient_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])

    dataset = []

    for patient in patient_dirs:
        pid = patient.name
        seg_path = patient / f"{pid}_seg.nii"

        # Skip patients with missing or non-standard segmentation files
        if not seg_path.exists():
            continue

        entry = {
            "image": [
                str(patient / f"{pid}_flair.nii"),
                str(patient / f"{pid}_t1.nii"),
                str(patient / f"{pid}_t1ce.nii"),
                str(patient / f"{pid}_t2.nii"),
            ],
            "label": str(seg_path),
        }

        dataset.append(entry)

    return dataset


if __name__ == "__main__":
    files = build_brats_file_list("data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")

    print("Total patients:", len(files))
    print("\nExample entry:\n")
    print(files[0])