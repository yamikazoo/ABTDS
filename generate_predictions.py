import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from monai.inferers import sliding_window_inference
from src.components.data_module import BraTSDataModule
from src.train import BraTSModel

def main():
    print("Loading model...")
    # Load the model from checkpoint
    checkpoint_path = "outputs/checkpoints/brats-epoch=97-val_mean_dice=0.7316.ckpt"
    model = BraTSModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Setting up data...")
    # Setup DataModule to get validation data
    data_dir = "data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    data_module = BraTSDataModule(data_dir=data_dir, batch_size=1)
    data_module.setup()
    val_loader = data_module.val_dataloader()

    # Create output directory
    os.makedirs("outputs/predictions", exist_ok=True)

    # Generate predictions for a few samples
    num_samples = 3
    
    print("Generating predictions...")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break
                
            images, labels = batch["image"].to(device), batch["label"].to(device)
            
            # Use sliding window inference for full volume
            outputs = sliding_window_inference(
                inputs=images, 
                roi_size=(96, 96, 96), 
                sw_batch_size=4, 
                predictor=model.model,
                overlap=0.5
            )

            # Process outputs (discretize)
            # outputs shape: [1, 4, H, W, D] -> [1, 4] logits
            val_outputs = [model.post_pred(j) for j in outputs]
            val_outputs = torch.stack(val_outputs) # [1, 4, H, W, D]
            
            # Find the slice with the maximum tumor area in the ground truth
            label_np = labels[0, 0].cpu().numpy() # [H, W, D]
            tumor_pixels_per_slice = (label_np > 0).sum(axis=(0, 1))
            best_slice = np.argmax(tumor_pixels_per_slice)
            
            # Select FLAIR channel (index 0)
            flair_slice = images[0, 0, :, :, best_slice].cpu().numpy()
            
            # Ground truth slice
            label_slice = label_np[:, :, best_slice]
            
            # val_outputs is one-hot [1, 4, H, W, D]
            # Convert to argmax shape [1, 1, H, W, D] for visualization
            pred_argmax = torch.argmax(val_outputs, dim=1, keepdim=True)
            pred_slice = pred_argmax[0, 0, :, :, best_slice].cpu().numpy()

            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(flair_slice, cmap="gray")
            axes[0].set_title(f"FLAIR MRI (Slice {best_slice})")
            axes[0].axis("off")
            
            axes[1].imshow(label_slice, cmap="viridis")
            axes[1].set_title("Ground Truth Mask")
            axes[1].axis("off")
            
            axes[2].imshow(pred_slice, cmap="viridis")
            axes[2].set_title("Predicted Mask")
            axes[2].axis("off")
            
            plt.tight_layout()
            out_file = f"outputs/predictions/sample_{i}.png"
            plt.savefig(out_file)
            plt.close()
            print(f"Saved prediction image to {out_file}")
            
if __name__ == "__main__":
    # Disable duplicate lib error on windows
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    main()
