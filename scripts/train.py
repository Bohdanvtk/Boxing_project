# scripts/train.py
import argparse
from boxing_project.utils.config import load_cfg
from boxing_project.tasks import get_task
from boxing_project.training.trainer import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to YAML (referee.yaml / participants.yaml)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    Task = get_task(cfg)
    train(cfg, Task)
