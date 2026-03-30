"""
Launch a W&B hyperparameter sweep for BraTS segmentation.

Usage:
    python sweep.py                         # create sweep and start 1 agent
    python sweep.py --count 5               # run 5 trials
    python sweep.py --sweep_id <id>         # join an existing sweep
"""

import argparse
import wandb
import yaml


def main():
    parser = argparse.ArgumentParser(description="Launch W&B Sweep")
    parser.add_argument("--sweep_config", type=str, default="config/sweep.yaml",
                        help="Path to sweep YAML config")
    parser.add_argument("--sweep_id", type=str, default=None,
                        help="Existing sweep ID to join (skip creation)")
    parser.add_argument("--project", type=str, default="brats-segmentation",
                        help="W&B project name")
    parser.add_argument("--count", type=int, default=1,
                        help="Number of sweep trials to run")
    args = parser.parse_args()

    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        with open(args.sweep_config, "r") as f:
            sweep_config = yaml.safe_load(f)
        sweep_id = wandb.sweep(sweep_config, project=args.project)
        print(f"Created sweep: {sweep_id}")

    wandb.agent(sweep_id, project=args.project, count=args.count)


if __name__ == "__main__":
    main()
