import torch
import pytorch_lightning as pl
from monai.networks.nets import SegResNet
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

class BraTSModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        
        # SegResNet is highly recommended for BraTS over standard UNet
        self.model = SegResNet(
            spatial_dims=3,
            in_channels=4,  # 4 MRI modalities
            out_channels=4, # 4 classes: Background, NCR, ED, ET
        )
        
        # DiceCELoss tackles class imbalance beautifully
        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        # include_background=False because we only care about tumor regions
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        
        # Transforms to turn logits/raw labels into one-hot binary maps
        self.post_pred = AsDiscrete(argmax=True, to_onehot=4)
        self.post_label = AsDiscrete(to_onehot=4)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        # Using ReduceLROnPlateau to dynamically reduce LR when validation loss stops improving
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }

    def training_step(self, batch, batch_idx):
        # Yami's Training Loop
        images, labels = batch["image"], batch["label"]
        outputs = self.forward(images)
        
        loss = self.loss_function(outputs, labels)
        
        # Log training loss for Tallal's W&B dashboard
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # The Validation Strategy
        images, labels = batch["image"], batch["label"]
        
        # 1. Sliding Window Inference
        # We process the full brain volume using 96x96x96 overlapping patches
        outputs = sliding_window_inference(
            inputs=images, 
            roi_size=(96, 96, 96), 
            sw_batch_size=4, 
            predictor=self.model,
            overlap=0.5 # 50% overlap prevents seams between patches
        )

        # 2. Compute Validation Loss
        val_loss = self.loss_function(outputs, labels)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)

        # 3. Discretize for Metrics
        # Convert outputs (logits) and labels into one-hot format
        val_outputs = [self.post_pred(i) for i in outputs]
        val_labels = [self.post_label(i) for i in labels]

        # 4. Feed to Metric Buffer
        # (Note: In a more advanced version, you'd map these to WT, TC, and ET here)
        self.dice_metric(y_pred=val_outputs, y=val_labels)

    def on_validation_epoch_end(self):
        # Aggregate and Log Metrics
        # PyTorch Lightning calls this automatically when the validation epoch finishes
        # Calculate the mean Dice score across all patients in the validation set
        mean_val_dice = self.dice_metric.aggregate().item()
        
        # Reset the metric buffer for the next epoch
        self.dice_metric.reset()
        
        # Log the final metric so Tallal can track it
        self.log("val_mean_dice", mean_val_dice, prog_bar=True, sync_dist=True)