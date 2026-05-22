# Run: python src/train.py --config experiments/densenet121_seed42_fold0.yaml

from pathlib import Path
import argparse
import yaml

from torch.utils.data import DataLoader

from config import PROJECT_ROOT
from data import prepare_data
from datasets import CXRDataset, build_datasets
from transforms import build_transforms

def load_run_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.is_file():
        config_path = PROJECT_ROOT / config_path
    
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    run_dir = (PROJECT_ROOT / cfg["output_dir"] / cfg["run_name"]).resolve()
    cfg["run_dir"] = str(run_dir)

    return cfg

def initialize_run_dir(cfg: dict) -> Path:
    run_dir = Path(cfg["run_dir"])
    config_out_path = run_dir / "config.yaml"

    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory already exists and is not empty: {run_dir}"
        )
    
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    with config_out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    return run_dir

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_run_config(args.config)

    print("=== Run Config ===")
    print(f"Run: {cfg['run_name']}")
    print(f"Model: {cfg['model_name']}")
    print(f"Backbone LR: {cfg['optimizer']['backbone_lr']}")
    print(f"Rotation Degree: {cfg['augmentation']['train']['rotation_deg']}")

    bundle = prepare_data(seed=cfg["seed"])

    print("=== Data Summary ===")
    print(f"Train rows: {len(bundle.train_df)}")
    print(f"Val rows: {len(bundle.val_df)}")
    print(f"Test rows: {len(bundle.test_df)}")
    print(f"Num classes: {len(bundle.class_names)}")
    print(f"First classes: {bundle.class_names[:5]}")
    print(f"Num target columns: {len(bundle.target_columns)}")
    print(f"Pos weight shape: {bundle.pos_weights.shape}")

    initialize_run_dir(cfg)

    transform_dict = build_transforms(cfg)

    #DataLoader Smoke Test
    datasets = build_datasets(
        bundle,
        train_transform=transform_dict["train"],
        eval_transform=transform_dict["eval"],
    )
    
    train_loader = DataLoader(datasets["train"], batch_size=cfg['batch_size'], shuffle=True)

    batch = next(iter(train_loader))
    print("=== Batch Summary ===")
    print(batch["image"].shape)
    print(batch["target"].shape)
    print(batch["image_id"][:2])

if __name__ == "__main__":
    main()