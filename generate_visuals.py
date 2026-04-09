"""
Generate prediction visualizations from a trained BraTS model checkpoint.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python generate_visuals.py

Produces side-by-side images: MRI slice | Ground Truth | Prediction
Saved to Figures/ directory for the Overleaf report.
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete

from src.train import BraTSModel
from src.components.data_module import BraTSDataModule


def find_best_slice(label_vol):
    """Find the axial slice with the most tumor voxels."""
    tumor_mask = (label_vol > 0).astype(np.float32)
    slice_sums = tumor_mask.sum(axis=(0, 1))
    return int(np.argmax(slice_sums))


def main():
    os.makedirs("Figures", exist_ok=True)

    # Load trained model
    ckpt_path = "outputs/checkpoints/brats-epoch=93-val_mean_dice=0.7317.ckpt"
    model = BraTSModel.load_from_checkpoint(ckpt_path)
    model.eval()
    model.freeze()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    # Load a few validation samples
    datamodule = BraTSDataModule(
        data_dir="data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
        batch_size=1,
        num_workers=0,
    )
    datamodule.setup()

    post_pred = AsDiscrete(argmax=True)

    # Generate visuals for 3 sample patients
    num_samples = 3
    for idx in range(num_samples):
        sample = datamodule.val_dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        label = sample["label"].numpy()

        # Run inference
        with torch.no_grad():
            output = sliding_window_inference(
                inputs=image,
                roi_size=(96, 96, 96),
                sw_batch_size=4,
                predictor=model,
                overlap=0.5,
            )
            pred = post_pred(output[0]).cpu().numpy()

        image_np = sample["image"].numpy()
        label_np = label[0]  # remove channel dim
        pred_np = pred[0]    # remove channel dim

        # Find the slice with the most tumor
        best_slice = find_best_slice(label_np)

        # Get the FLAIR modality (index 0) for display
        flair_slice = image_np[0, :, :, best_slice]
        gt_slice = label_np[:, :, best_slice]
        pred_slice = pred_np[:, :, best_slice]

        # Color map for tumor regions: 1=NCR (red), 2=ED (green), 3=ET (yellow)
        colors = {1: [1, 0, 0], 2: [0, 1, 0], 3: [1, 1, 0]}

        def overlay_mask(mri, mask, alpha=0.5):
            """Overlay colored segmentation mask on grayscale MRI."""
            mri_norm = (mri - mri.min()) / (mri.max() - mri.min() + 1e-8)
            rgb = np.stack([mri_norm] * 3, axis=-1)
            for label_val, color in colors.items():
                region = mask == label_val
                for c in range(3):
                    rgb[:, :, c] = np.where(region, rgb[:, :, c] * (1 - alpha) + color[c] * alpha, rgb[:, :, c])
            return rgb

        gt_overlay = overlay_mask(flair_slice, gt_slice)
        pred_overlay = overlay_mask(flair_slice, pred_slice)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(flair_slice.T, cmap="gray", origin="lower")
        axes[0].set_title("FLAIR MRI", fontsize=14)
        axes[0].axis("off")

        axes[1].imshow(gt_overlay.transpose(1, 0, 2), origin="lower")
        axes[1].set_title("Ground Truth", fontsize=14)
        axes[1].axis("off")

        axes[2].imshow(pred_overlay.transpose(1, 0, 2), origin="lower")
        axes[2].set_title("Prediction (Dice: 0.7317)", fontsize=14)
        axes[2].axis("off")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="red", label="NCR (Necrotic Core)"),
            Patch(facecolor="green", label="ED (Edema)"),
            Patch(facecolor="yellow", label="ET (Enhancing Tumor)"),
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=11, frameon=False)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        save_path = f"Figures/prediction_sample_{idx + 1}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    print("\nDone! Upload the Figures/ folder to Overleaf.")


if __name__ == "__main__":
    main()
