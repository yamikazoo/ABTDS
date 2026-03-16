import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.components.data_module import BraTSDataModule
from src.train import BraTSModel


def main(args):
    pl.seed_everything(42)

    datamodule = BraTSDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = BraTSModel(learning_rate=args.lr)

    checkpoint_callback = ModelCheckpoint(
        dirpath="outputs/checkpoints",
        monitor="val_mean_dice",
        mode="max",
        save_top_k=3,
        filename="brats-{epoch:02d}-{val_mean_dice:.4f}"
    )

    logger = TensorBoardLogger(save_dir="outputs/logs", name="brats_baseline")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto", 
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        precision="16-mixed" if args.amp else "32-true",
        fast_dev_run=args.fast_dev_run,
        overfit_batches=args.overfit_batches,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BraTS 2020 Segmentation Model")
    parser.add_argument("--data_dir", type=str, default="data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision")
    
    # Testing/Debugging flags
    parser.add_argument("--fast_dev_run", action="store_true", help="Run 1 train and 1 val batch to catch bugs")
    parser.add_argument("--overfit_batches", type=float, default=0.0, help="Overfit on a subset of data (e.g., 0.01 or 1)")
    
    args = parser.parse_args()
    main(args)