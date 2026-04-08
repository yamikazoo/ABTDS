import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import label


def remove_small_components(mask: np.ndarray, min_size: int, connectivity: int = 3) -> np.ndarray:
    """Remove connected components smaller than min_size from a binary mask."""
    if min_size <= 0:
        return mask

    structure = np.ones((3, 3, 3), dtype=np.uint8) if connectivity == 3 else None
    labeled, num_features = label(mask.astype(np.uint8), structure=structure)
    if num_features == 0:
        return mask

    counts = np.bincount(labeled.ravel())
    keep = counts >= min_size
    keep[0] = False
    return keep[labeled]


def clean_prediction(
    pred: np.ndarray,
    min_size: int,
    classes: list[int] | None = None,
    connectivity: int = 3,
) -> np.ndarray:
    """
    Remove tiny connected components from a label map prediction.

    Expects integer labels (e.g., BraTS: 0 background, 1/2/3 tumor classes).
    """
    cleaned = pred.copy()
    class_ids = classes if classes else [int(c) for c in np.unique(pred) if c != 0]

    for class_id in class_ids:
        class_mask = pred == class_id
        filtered = remove_small_components(class_mask, min_size=min_size, connectivity=connectivity)
        removed = class_mask & ~filtered
        cleaned[removed] = 0

    return cleaned


def visualize_cleanup(
    original: np.ndarray,
    cleaned: np.ndarray,
    save_path: Path,
    name: str,
) -> None:
    """Save a before/after/diff comparison PNG for the slice with the most changes."""
    diff = (original > 0).astype(int) - (cleaned > 0).astype(int)
    removed_per_slice = diff.sum(axis=(0, 1))
    best_slice = int(np.argmax(removed_per_slice))

    if removed_per_slice[best_slice] == 0:
        tumor_per_slice = (original > 0).sum(axis=(0, 1))
        best_slice = int(np.argmax(tumor_per_slice))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original[:, :, best_slice], cmap="viridis")
    axes[0].set_title("Before (Raw Prediction)")
    axes[0].axis("off")

    axes[1].imshow(cleaned[:, :, best_slice], cmap="viridis")
    axes[1].set_title("After (Cleaned)")
    axes[1].axis("off")

    axes[2].imshow(diff[:, :, best_slice], cmap="Reds")
    axes[2].set_title("Removed Voxels")
    axes[2].axis("off")

    fig.suptitle(f"{name}  (slice {best_slice})", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def iter_prediction_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.nii.gz"))
    files.extend(sorted(input_dir.glob("*.nii")))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process segmentation predictions using connected component analysis."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with predicted masks (.nii/.nii.gz)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save cleaned masks")
    parser.add_argument(
        "--min_size",
        type=int,
        default=50,
        help="Minimum voxel count for a connected component to be kept",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Class IDs to clean. Default: all non-background labels found in each volume.",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[1, 3],
        default=3,
        help="1 for face connectivity, 3 for full 3D neighborhood connectivity",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save before/after/diff comparison PNGs alongside the cleaned masks",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_files = iter_prediction_files(input_dir)
    if not pred_files:
        raise FileNotFoundError(f"No .nii or .nii.gz files found in {input_dir}")

    for pred_path in pred_files:
        img = nib.load(str(pred_path))
        pred = np.asarray(img.get_fdata()).astype(np.int16)
        cleaned = clean_prediction(
            pred=pred,
            min_size=args.min_size,
            classes=args.classes,
            connectivity=args.connectivity,
        )

        removed_count = int(np.sum((pred > 0) & (cleaned == 0)))

        out_img = nib.Nifti1Image(cleaned.astype(np.int16), img.affine, img.header)
        out_path = output_dir / pred_path.name
        nib.save(out_img, str(out_path))
        print(f"cleaned: {pred_path.name} -> {out_path}  ({removed_count} voxels removed)")

        if args.visualize:
            png_path = output_dir / f"{pred_path.name.split('.')[0]}_comparison.png"
            visualize_cleanup(pred, cleaned, png_path, pred_path.name)
            print(f"  visual: {png_path}")


if __name__ == "__main__":
    main()
