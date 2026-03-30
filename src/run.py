import argparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src.components.data_module import BraTSDataModule
from src.train import BraTSModel


def load_config(config_path):
    """Load hyperparameters from a YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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

    loggers = [TensorBoardLogger(save_dir="outputs/logs", name="brats_baseline")]

    if args.wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name,
            save_dir="outputs/logs",
            log_model=False,
        )
        wandb_logger.experiment.config.update(vars(args))
        loggers.append(wandb_logger)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        logger=loggers,
        callbacks=[checkpoint_callback],
        precision="16-mixed" if args.amp else "32-true",
        fast_dev_run=args.fast_dev_run,
        overfit_batches=args.overfit_batches,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BraTS 2020 Segmentation Model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision")

    # W&B flags
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="brats-segmentation", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")

    # Testing/Debugging flags
    parser.add_argument("--fast_dev_run", action="store_true", help="Run 1 train and 1 val batch to catch bugs")
    parser.add_argument("--overfit_batches", type=float, default=None)

    args = parser.parse_args()

    # Load defaults from YAML config, then override with any CLI args
    defaults = {
        "data_dir": "data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
        "batch_size": 1,
        "num_workers": 4,
        "lr": 1e-4,
        "epochs": 50,
        "overfit_batches": 0.0,
    }

    if args.config:
        yaml_config = load_config(args.config)
        defaults.update(yaml_config)

    # CLI args override YAML values (only if explicitly provided)
    for key, default_val in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, default_val)

    main(args)