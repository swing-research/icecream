from __future__ import annotations
import typer, yaml
from pathlib import Path
from typing import Optional, List, Dict, Any

from .train import train_model
from .predict import predict

from .utils.utils import split_tilt_series
from torch.profiler import profile, record_function, ProfilerActivity

app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False, help="Cryo-ET training & prediction CLI")


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
        config: Optional[Path] = typer.Option(None, help="(Optional) Path to the YAML config file."),

        # data
        tomo0: Optional[List[str]] = typer.Option(None, help="Path to the first tomogram."),
        tomo1: Optional[List[str]] = typer.Option(None, help="Path to the second tomogram."),
        mask: Optional[List[str]] = typer.Option(None,
                                            help="(Optional) Path to tomogram masks on the spatial domain, e.g. using Slabify. If empty, masks will be created similarly to IsoNet."),
        angles: Optional[List[str]] = typer.Option(None,
                                              help="(Optional) Path to the tilt angle files. Valid extensions include '.txt' and '.tlt'."),
        tilt_min: Optional[float] = typer.Option(None,
                                                 help="(Optional) Minimum tilt angle in degrees. Default is -60."),
        tilt_max: Optional[float] = typer.Option(None,
                                                 help="(Optional) Maximum tilt angle in degrees. Default is +60."),
        save_dir: Optional[Path] = typer.Option(None,
                                                help="(Optional) Path to the directory to save the trained model. Default is 'runs/default/'."),

        # training knobs
        batch_size: Optional[int] = typer.Option(None,
                                                 help="(Optional) Training batch size. Reduce if memory issue. Default is 8."),
        crop_size: Optional[int] = typer.Option(None,
                                                help="(Optional) Crop size of subtomograms for prediction. Default is 72x72."),
        eq_weight: Optional[float] = typer.Option(None,
                                              help="(Optional) Equivariant regularization weight. Increase to get more regularization. Default is 2."),
        iterations: Optional[int] = typer.Option(None, help="(Optional) Number of iterations. Default is 50000."),
        save_n_iterations: Optional[int] = typer.Option(None,
                                                        help="(Optional) Checkpoint for the model every N iterations. Default is 5000."),
        compute_avg_loss_n_iterations: Optional[int] = typer.Option(None,
                                                                    help="(Optional) Average loss every N iterations. Default is 1000."),
        save_tomo_n_iterations: Optional[int] = typer.Option(None,
                                                             help="(Optional) Run the tomogram reconstruction every N iterations. One reconstruction might take several minutes. Default is None."),
        pretrain_path: Optional[Path] = typer.Option(None,
                                                     help="(Optional) Pretrained model path (location to .pt file)."),
):
    cfg = load_defaults()
    if config:
        cfg = deep_update(cfg, load_yaml(config))

    cli_updates: Dict[str, Any] = {"data": {}, "train_params": {}, "predict_params": {} }
    if tomo0: cli_updates["data"]["tomo0"] = tomo0
    if tomo1: cli_updates["data"]["tomo1"] = tomo1
    if mask: cli_updates["data"]["mask"] = mask
    if angles: cli_updates["data"]["angles"] = angles
    if tilt_min is not None: cli_updates["data"]["tilt_min"] = tilt_min
    if tilt_max is not None: cli_updates["data"]["tilt_max"] = tilt_max
    if save_dir: cli_updates["data"]["save_dir"] = str(save_dir)

    if batch_size is not None: cli_updates["train_params"]["batch_size"] = batch_size
    if crop_size is not None:
        cli_updates["train_params"]["crop_size"] = crop_size
        cli_updates["predict_params"]["stride"] = crop_size//2
    if iterations is not None: cli_updates["train_params"]["iterations"] = iterations
    if save_n_iterations is not None: cli_updates["train_params"]["save_n_iterations"] = save_n_iterations
    if eq_weight is not None: cli_updates["train_params"]["eq_weight"] = eq_weight
    if compute_avg_loss_n_iterations is not None:
        cli_updates["train_params"]["compute_avg_loss_n_iterations"] = compute_avg_loss_n_iterations
    if save_tomo_n_iterations is not None:
        cli_updates["train_params"]["save_tomo_n_iterations"] = save_tomo_n_iterations

    if pretrain_path is not None:
        cli_updates["pretrain_params"] = {
            "use_pretrain": True,
            "model_path": str(pretrain_path)
        }

    cfg = deep_update(cfg, cli_updates)

    need = ["data.tomo0", "data.tomo1", "data.save_dir"]
    if not cfg["data"].get("angles"):
        need += ["data.tilt_min", "data.tilt_max"]
    require(cfg, need)

    if cfg['debug']['profiling']:
        with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,  # optional: track tensor shapes
                profile_memory=True,  # optional: track memory usage
                with_stack=False,  # set True for call stack traces
        ) as prof:
            with record_function("model_inference"):
                train_model(cfg)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=500))
    else:
        train_model(cfg)


# ---------- subcommand: predict ----------
@app.command("predict")
def cli_predict(
        # necessary config
        config: Optional[Path] = typer.Option(None, help="(Optional) Path to the YAML config file."),

        # data
        tomo0: Optional[List[str]] = typer.Option(None, help="Path to the first tomogram."),
        tomo1: Optional[List[str]] = typer.Option(None, help="Path to the second tomogram."),
        angles: Optional[List[str]] = typer.Option(None,
                                              help="(Optional) Path to the tilt angle file. Valid extension include '.txt' and '.tlt'."),
        tilt_min: Optional[float] = typer.Option(None,
                                                 help="(Optional) Minimum tilt angle in degrees. Default is -60."),
        tilt_max: Optional[float] = typer.Option(None,
                                                 help="(Optional) Maximum tilt angle in degrees. Default is +60."),
        save_dir: Optional[Path] = typer.Option(None,
                                                help="(Optional) Path to the directory where the trained model is saved. Default is 'runs/default/'."),
        save_dir_reconstructions: Optional[Path] = typer.Option(None,
                                                help="(Optional) Path to the save the reconstructions. Defaults is same as save_dire."),

        # optional overrides
        batch_size: Optional[int] = typer.Option(None,
                                                 help="(Optional) Training batch size. Reduce if memory issue. Default is 8."),
        crop_size: Optional[int] = typer.Option(None,
                                                help="(Optional) Crop size of subtomograms for prediction. Default is 72x72."),
        iter_load: int = typer.Option(-1, help="Iteration to load (default: latest in save_dir/model)"),
):
    cfg = load_defaults()
    if config:
        cfg = deep_update(cfg, load_yaml(config))

    cli_updates: Dict[str, Any] = {"data": {}, "predict_params": {}, "train_params": {}}
    if save_dir: cli_updates["data"]["save_dir"] = str(save_dir)
    if tomo0: cli_updates["data"]["tomo0"] = tomo0
    if tomo1: cli_updates["data"]["tomo1"] = tomo1
    if angles: cli_updates["data"]["angles"] = angles
    if tilt_min is not None: cli_updates["data"]["tilt_min"] = tilt_min
    if tilt_max is not None: cli_updates["data"]["tilt_max"] = tilt_max
    if iter_load is not None: cli_updates["predict_params"]["iter_load"] = iter_load
    if save_dir_reconstructions is not None:
        cli_updates["predict_params"]["save_dir_reconstructions"] = str(save_dir_reconstructions)
    if batch_size is not None:
        cli_updates["predict_params"]["batch_size"] = batch_size
    if crop_size is not None:
        cli_updates["train_params"]["crop_size"] = crop_size
        cli_updates["predict_params"]["stride"] = crop_size//2

    cfg = deep_update(cfg, cli_updates)

    need = ["data.tomo0", "data.save_dir"]
    if not cfg["data"].get("angles"):
        need += ["data.tilt_min", "data.tilt_max"]
    require(cfg, need)

    if cfg['debug']['profiling']:
        with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,  # optional: track tensor shapes
                profile_memory=True,  # optional: track memory usage
                with_stack=False,  # set True for call stack traces
        ) as prof:
            with record_function("model_inference"):
                predict(cfg)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=500))
    else:
        predict(cfg)


# ---------- subcommand: predict ----------
@app.command("split-tilt-series")
def cli_split_tilt_series(
        # data
        path_ts: Path = typer.Option(..., help="Path to the tilt-series to split."),
        angles: Optional[Path] = typer.Option(None,
                                              help="(Optional) Path to the tilt angle file. Valid extension include '.txt' and '.tlt'.", ),
        tilt_min: Optional[float] = typer.Option(None,
                                                 help="(Optional) Minimum tilt angle in degrees."),
        tilt_max: Optional[float] = typer.Option(None,
                                                 help="(Optional) Maximum tilt angle in degrees."),
        save_dir: Optional[Path] = typer.Option(None,
                                                help="(Optional) Path to the directory to save the trained model. Default is the directory of the tilt-series."),
):
    split_tilt_series(str(path_ts), str(angles), tilt_min, tilt_max, save_dir)


if __name__ == "__main__":
    app()
