from __future__ import annotations
import typer, yaml
from pathlib import Path
from typing import Optional, List, Dict, Any

from .train import train_model
from .predict import predict  # put your predict() in src/cryoei/predict.py

app = typer.Typer(add_completion=False, help="Cryo-ET training & prediction CLI")

# ---------- helpers ----------
def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def deep_update(base: dict, updates: dict) -> dict:
    out = dict(base)
    for k, v in (updates or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        elif v is not None:
            out[k] = v
    return out

def load_defaults() -> dict:
    default_path = Path(__file__).with_name("defaults.yaml")
    return load_yaml(default_path)

def require(cfg: dict, keys: List[str]):
    missing = []
    for dotted in keys:
        node = cfg
        for part in dotted.split("."):
            node = None if node is None else node.get(part)
        if node in (None, [], ""):
            missing.append(dotted)
    if missing:
        typer.secho(f"Missing required option(s): {', '.join(missing)}", fg="red")
        raise typer.Exit(code=2)

# ---------- subcommand: train ----------
@app.command("train")
def cli_train(
    config: Optional[Path] = typer.Option(None, help="YAML config file"),

    # data
    tomo0: Optional[List[str]] = typer.Option(None, help="Volume set 0"),
    tomo1: Optional[List[str]] = typer.Option(None, help="Volume set 1"),
    mask:  Optional[Path]      = typer.Option(None, help="Mask file"),
    angles: Optional[Path]     = typer.Option(None, help="Angles file"),
    tilt_min: Optional[float]  = typer.Option(None),
    tilt_max: Optional[float]  = typer.Option(None),
    save_dir: Optional[Path]   = typer.Option(None),

    # training knobs
    batch_size: Optional[int] = typer.Option(None, help="Training batch size"),
    scale: Optional[float] = typer.Option(None, help="Scaling factor"),
    epochs: Optional[int] = typer.Option(None, help="Number of epochs"),
    save_n_epochs: Optional[int] = typer.Option(None, help="Checkpoint every N epochs"),
    compute_avg_loss: Optional[int] = typer.Option(None, help="Average loss every N epochs"),
):
    cfg = load_defaults()
    if config:
        cfg = deep_update(cfg, load_yaml(config))

    cli_updates: Dict[str, Any] = {"data": {}, "train_params": {}}
    if tomo0: cli_updates["data"]["tomo0_files"] = tomo0
    if tomo1: cli_updates["data"]["tomo1_files"] = tomo1
    if mask: cli_updates["data"]["mask_file"] = str(mask)
    if angles: cli_updates["data"]["angle_file"] = str(angles)
    if tilt_min is not None: cli_updates["data"]["tilt_min"] = tilt_min
    if tilt_max is not None: cli_updates["data"]["tilt_max"] = tilt_max
    if save_dir: cli_updates["data"]["save_dir"] = str(save_dir)

    if batch_size is not None: cli_updates["train_params"]["batch_size"] = batch_size
    if epochs is not None: cli_updates["train_params"]["epochs"] = epochs
    if save_n_epochs is not None: cli_updates["train_params"]["save_n_epochs"] = save_n_epochs
    if scale is not None: cli_updates["train_params"]["scale"] = scale
    if compute_avg_loss is not None:
        cli_updates["train_params"]["compute_avg_loss_n_epochs"] = compute_avg_loss

    cfg = deep_update(cfg, cli_updates)

    need = ["data.tomo0_files", "data.tomo1_files", "data.save_dir"]
    if not cfg["data"].get("angle_file"):
        need += ["data.tilt_min", "data.tilt_max"]
    require(cfg, need)

    train_model(cfg)

# ---------- subcommand: predict ----------
@app.command("predict")
def cli_predict(
    config: Optional[Path] = typer.Option(None, help="YAML config file"),
    # optional overrides
    epoch: int = typer.Option(-1, help="Epoch to load (default: latest in save_dir/model)"),
    crop_size: Optional[int] = typer.Option(None, help="Crop size for prediction"),
    batch_size: Optional[int] = typer.Option(None, help="Batch size for prediction"),

    # (allow data overrides too, in case user points to a different exp)
    save_dir: Optional[Path] = typer.Option(None, help="Output/experiment directory"),
    save_name: Optional[str] = typer.Option(None, help="Output volume name"),
    tomo0: Optional[List[str]] = typer.Option(None, help="Volume set 0"),
    tomo1: Optional[List[str]] = typer.Option(None, help="Volume set 1"),
    mask:  Optional[Path] = typer.Option(None, help="Mask file"),
    angles: Optional[Path] = typer.Option(None, help="Angles file"),
    tilt_min: Optional[float] = typer.Option(None),
    tilt_max: Optional[float] = typer.Option(None),
):
    cfg = load_defaults()
    if config:
        cfg = deep_update(cfg, load_yaml(config))

    cli_updates: Dict[str, Any] = {"data": {}, "predict_params": {}}
    if save_dir: cli_updates["data"]["save_dir"] = str(save_dir)
    if tomo0: cli_updates["data"]["tomo0_files"] = tomo0
    if tomo1: cli_updates["data"]["tomo1_files"] = tomo1
    if mask: cli_updates["data"]["mask_file"] = str(mask)
    if angles: cli_updates["data"]["angle_file"] = str(angles)
    if tilt_min is not None: cli_updates["data"]["tilt_min"] = tilt_min
    if tilt_max is not None: cli_updates["data"]["tilt_max"] = tilt_max

    if batch_size is not None:
        cli_updates["predict_params"]["batch_size"] = batch_size
    if crop_size is not None:
        cli_updates["predict_params"]["crop_size"] = crop_size
    

    cfg = deep_update(cfg, cli_updates)

    need = ["data.tomo0_files", "data.tomo1_files", "data.save_dir"]
    if not cfg["data"].get("angle_file"):
        need += ["data.tilt_min", "data.tilt_max"]
    require(cfg, need)

    # your predict() signature: predict(config_yaml, epoch=-1, crop_size=None, batch_size=0)
    predict(cfg, epoch=epoch, crop_size=crop_size, batch_size=(batch_size or 0), save_name=save_name)

if __name__ == "__main__":
    app()