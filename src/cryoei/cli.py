# src/cryoei/cli.py
import typer, yaml
from pathlib import Path
from typing import Optional, List, Any, Dict
from .train import train_model

app = typer.Typer(add_completion=False)

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

@app.command("run")
def run(
    config: Optional[Path] = typer.Option(None, help="YAML config file"),

    # data-related overrides
    tomo0: Optional[List[str]] = typer.Option(None, help="Volume set 0"),
    tomo1: Optional[List[str]] = typer.Option(None, help="Volume set 1"),
    mask: Optional[Path] = typer.Option(None, help="Mask file"),
    angles: Optional[Path] = typer.Option(None, help="Angle file"),
    tilt_min: Optional[float] = typer.Option(None, help="Minimum tilt angle (in degrees)"),
    tilt_max: Optional[float] = typer.Option(None, help="Maximum tilt angle (in degrees)"),
    save_dir: Optional[Path] = typer.Option(None, help="Directory to save outputs"),

    # training knobs exposed in CLI
    batch_size: Optional[int] = typer.Option(None, help="Training batch size"),
    epochs: Optional[int] = typer.Option(None, help="Number of training epochs"),
    save_n_epochs: Optional[int] = typer.Option(None, help="Save model every N epochs"),
    compute_avg_loss: Optional[int] = typer.Option(None, help="Compute avg loss every N epochs"),
):
    # 1) load defaults.yaml
    default_path = Path(__file__).with_name("defaults.yaml")
    cfg = load_yaml(default_path)

    # 2) overlay user config
    if config:
        cfg = deep_update(cfg, load_yaml(config))

    # 3) overlay CLI overrides
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
    if compute_avg_loss is not None: cli_updates["train_params"]["compute_avg_loss_n_epochs"] = compute_avg_loss

    cfg = deep_update(cfg, cli_updates)

    # sanity check: must have tomo0, tomo1, save_dir, and either angle_file or tilt_min/max
    missing = []
    for key in ["data.tomo0_files", "data.tomo1_files", "data.save_dir"]:
        node = cfg
        for part in key.split("."):
            node = node.get(part) if node else None
        if not node:
            missing.append(key)
    if not cfg["data"].get("angle_file"):
        if cfg["data"].get("tilt_min") is None or cfg["data"].get("tilt_max") is None:
            missing += ["data.angle_file OR (data.tilt_min & data.tilt_max)"]
    if missing:
        typer.secho(f"Missing required options: {', '.join(missing)}", fg="red")
        raise typer.Exit(code=2)

    train_model(cfg)
