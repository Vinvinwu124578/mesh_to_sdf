from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import (
    collect_npz_files,
    install_amp_safe_structured_oracle_patch,
    parse_index_list,
    save_json,
    set_seed,
    split_object_files,
)
from tactistruct_structured_oracle.dataset import StructuredTouchPoseDataset
from tactistruct_structured_oracle.losses import compute_structured_oracle_losses
from tactistruct_structured_oracle.model import StructuredTouchOracleSystem

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


install_amp_safe_structured_oracle_patch()


DEFAULT_OUTPUT_DIR = "outputs/structured_oracle_mujoco_coverage_no_pose"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the structured tactile oracle on coverage-aware MuJoCo structured patch data "
            "without any pose augmentation."
        )
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--touch-view-indices", type=str, default=None)
    parser.add_argument("--num-surface-points", type=int, default=4096)
    parser.add_argument("--num-query-points", type=int, default=6144)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache-train-in-memory", action="store_true")
    parser.add_argument("--cache-val-in-memory", action="store_true")
    parser.add_argument("--enable-wandb", action="store_true")
    return parser.parse_args()


def maybe_init_wandb(config: dict, output_dir: Path):
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb logging is enabled but the 'wandb' package is not installed."
        ) from exc
    return wandb.init(
        project=wandb_cfg.get("project", "structured-oracle-mujoco-coverage-no-pose"),
        name=wandb_cfg.get("name"),
        mode=wandb_cfg.get("mode", "online"),
        tags=wandb_cfg.get("tags"),
        dir=str(output_dir),
        config=config,
    )


def build_dataset_cfg(
    data_dir: Path,
    object_filenames: list[str],
    touch_view_indices: list[int] | None,
    seed: int,
    num_surface_points: int,
    num_query_points: int,
    cache_in_memory: bool,
) -> dict:
    return {
        "name": "structured_touch",
        "root": str(data_dir.resolve()),
        "split": ".",
        "object_filenames": object_filenames,
        "recursive": True,
        "touch_view_indices": touch_view_indices,
        "num_surface_points": int(num_surface_points),
        "num_query_points": int(num_query_points),
        "dynamic_sampling": True,
        "near_surface_ratio": 0.6,
        "near_surface_threshold": 0.015,
        "negative_sample_ratio": 0.6,
        "seed": int(seed),
        "cache_in_memory": bool(cache_in_memory),
    }


def build_config(args: argparse.Namespace) -> tuple[dict, dict[str, list[str]]]:
    data_dir = Path(args.data_dir).resolve()
    files = collect_npz_files(data_dir)
    touch_indices = parse_index_list(args.touch_view_indices)
    train_files, val_files, test_files = split_object_files(
        files=files,
        data_dir=data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir).resolve()
    config = {
        "seed": int(args.seed),
        "dataset": {
            "train": build_dataset_cfg(
                data_dir=data_dir,
                object_filenames=train_files,
                touch_view_indices=touch_indices,
                seed=args.seed,
                num_surface_points=args.num_surface_points,
                num_query_points=args.num_query_points,
                cache_in_memory=bool(args.cache_train_in_memory),
            ),
        },
        "model": {
            "patch_encoder": {
                "token_dim": 256,
                "hidden_dim": 256,
                "point_hidden_dim": 128,
                "max_rounds": 16,
            },
            "round_fusion": {
                "hidden_dim": 256,
            },
            "global_encoder": {
                "hidden_dim": 256,
                "position_bands": 6,
                "num_layers": 2,
                "num_heads": 4,
                "dropout": 0.0,
            },
            "decoder": {
                "hidden_dim": 256,
                "position_bands": 8,
                "finger_position_bands": 6,
                "num_heads": 4,
            },
            "surface_query_samples": 2048,
        },
        "loss": {
            "sdf": 1.0,
            "surface": 0.25,
            "touch_zero": 0.5,
            "latent_l2": 1e-4,
            "center_alignment": 0.0,
            "eikonal": 0.0,
        },
        "train": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "prefetch_factor": int(args.prefetch_factor),
            "val_every": int(args.val_every),
            "amp": {"enabled": not bool(args.disable_amp)},
            "progress": {"enabled": not bool(args.disable_progress)},
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "device": str(args.device),
            "output_dir": str(output_dir),
            "pose_augmentation": False,
        },
        "wandb": {
            "enabled": bool(args.enable_wandb),
            "project": "structured-oracle-mujoco-coverage-no-pose",
            "name": output_dir.name,
            "mode": "online",
            "tags": ["structured-oracle", "mujoco", "coverage-aware", "no-pose-aug"],
        },
    }
    if val_files:
        config["dataset"]["val"] = build_dataset_cfg(
            data_dir=data_dir,
            object_filenames=val_files,
            touch_view_indices=touch_indices,
            seed=args.seed + 1000,
            num_surface_points=args.num_surface_points,
            num_query_points=args.num_query_points,
            cache_in_memory=bool(args.cache_val_in_memory),
        )
    if test_files:
        config["dataset"]["test"] = build_dataset_cfg(
            data_dir=data_dir,
            object_filenames=test_files,
            touch_view_indices=touch_indices,
            seed=args.seed + 2000,
            num_surface_points=args.num_surface_points,
            num_query_points=args.num_query_points,
            cache_in_memory=False,
        )
    split_summary = {"train": train_files, "val": val_files, "test": test_files}
    return config, split_summary


def build_dataset(cfg: dict) -> torch.utils.data.Dataset:
    params = {key: value for key, value in cfg.items() if key != "name"}
    return StructuredTouchPoseDataset(**params)


def make_loader_kwargs(device: torch.device, num_workers: int, prefetch_factor: int) -> dict:
    kwargs = {
        "num_workers": num_workers,
        "drop_last": False,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved = {}
    non_blocking = device.type == "cuda"
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=non_blocking) if torch.is_tensor(value) else value
    return moved


def sanitize_loss_dict(losses: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    sanitized: dict[str, torch.Tensor] = {}
    for key, value in losses.items():
        if torch.is_tensor(value):
            sanitized[key] = value if torch.isfinite(value).all() else torch.zeros((), device=device)
        else:
            sanitized[key] = value
    return sanitized


def summarise_metrics(metrics: dict[str, float], batches: int) -> dict[str, float]:
    return {key: value / max(batches, 1) for key, value in metrics.items()}


def save_checkpoint(
    path: Path,
    model: StructuredTouchOracleSystem,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    config: dict,
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "config": config,
            "epoch": int(epoch),
        },
        path,
    )


def run_epoch(
    model: StructuredTouchOracleSystem,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    loss_cfg: dict,
    device: torch.device,
    grad_clip: float | None,
    train_mode: bool,
    epoch: int,
    total_epochs: int,
    show_progress: bool,
    amp_enabled: bool,
) -> dict[str, float]:
    metrics = defaultdict(float)
    model.train(mode=train_mode)
    iterator = loader
    progress_bar = None
    if show_progress and tqdm is not None:
        phase = "train" if train_mode else "val"
        progress_bar = tqdm(
            loader,
            desc=f"{phase} {epoch:03d}/{total_epochs:03d}",
            leave=False,
            dynamic_ncols=True,
        )
        iterator = progress_bar

    for batch_index, batch in enumerate(iterator, start=1):
        batch = move_to_device(batch, device)
        if float(loss_cfg.get("eikonal", 0.0)) > 0.0 and "query_points" in batch:
            batch["query_points"] = batch["query_points"].clone().detach().requires_grad_(True)

        with torch.set_grad_enabled(train_mode):
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp_enabled and device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                outputs = model(batch)
                losses = compute_structured_oracle_losses(batch, outputs, loss_cfg)
                total = losses["total"]

            if train_mode and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and amp_enabled and device.type == "cuda":
                    scaler.scale(total).backward()
                    if grad_clip is not None and grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total.backward()
                    if grad_clip is not None and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

        losses = sanitize_loss_dict(losses, device=device)
        for key, value in losses.items():
            metrics[key] += float(value.detach().item())
        if progress_bar is not None:
            progress_bar.set_postfix(loss=f"{metrics['total'] / batch_index:.4f}")

    if progress_bar is not None:
        progress_bar.close()
    return summarise_metrics(metrics, len(loader))


def main() -> None:
    args = parse_args()
    config, split_summary = build_config(args)
    output_dir = Path(config["train"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_json(
        output_dir / "structured_oracle_mujoco_coverage_no_pose_config.json",
        config,
    )
    save_json(output_dir / "split_summary.json", split_summary)
    print(f"saved config to {config_path}")
    print(
        f"split summary | train={len(split_summary['train'])} "
        f"val={len(split_summary['val'])} test={len(split_summary['test'])}"
    )

    set_seed(int(config["seed"]))
    device = torch.device(config["train"]["device"])

    train_dataset = build_dataset(config["dataset"]["train"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        **make_loader_kwargs(
            device=device,
            num_workers=int(config["train"]["num_workers"]),
            prefetch_factor=int(config["train"]["prefetch_factor"]),
        ),
    )
    val_loader = None
    if "val" in config["dataset"]:
        val_dataset = build_dataset(config["dataset"]["val"])
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(config["train"]["batch_size"]),
            shuffle=False,
            **make_loader_kwargs(
                device=device,
                num_workers=int(config["train"]["num_workers"]),
                prefetch_factor=int(config["train"]["prefetch_factor"]),
            ),
        )

    model_cfg = config["model"]
    model = StructuredTouchOracleSystem(
        patch_encoder_cfg=model_cfg.get("patch_encoder", {}),
        round_fusion_cfg=model_cfg.get("round_fusion", {}),
        global_encoder_cfg=model_cfg.get("global_encoder", {}),
        decoder_cfg=model_cfg.get("decoder", {}),
        surface_query_samples=int(model_cfg.get("surface_query_samples", 2048)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    amp_enabled = bool(config["train"].get("amp", {}).get("enabled", False))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda")
    wandb_run = maybe_init_wandb(config, output_dir)

    best_val = float("inf")
    epochs = int(config["train"]["epochs"])
    grad_clip = float(config["train"]["grad_clip"])
    val_every = int(config["train"]["val_every"])
    show_progress = bool(config["train"].get("progress", {}).get("enabled", True))
    history: list[dict[str, float | int | str]] = []

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            loss_cfg=config["loss"],
            device=device,
            grad_clip=grad_clip,
            train_mode=True,
            epoch=epoch,
            total_epochs=epochs,
            show_progress=show_progress,
            amp_enabled=amp_enabled,
        )
        print(
            f"epoch {epoch:03d} | train total={train_metrics['total']:.4f} "
            f"sdf={train_metrics.get('shape_sdf', 0.0):.4f}"
        )
        history.append({"epoch": epoch, "split": "train", **train_metrics})
        save_checkpoint(output_dir / "latest.pt", model, optimizer, scaler, config, epoch)
        if wandb_run is not None:
            wandb_run.log({f"train/{key}": value for key, value in train_metrics.items()}, step=epoch)

        if val_loader is None or epoch % val_every != 0:
            continue

        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                scaler=None,
                loss_cfg=config["loss"],
                device=device,
                grad_clip=None,
                train_mode=False,
                epoch=epoch,
                total_epochs=epochs,
                show_progress=show_progress,
                amp_enabled=amp_enabled,
            )
        print(
            f"epoch {epoch:03d} | val   total={val_metrics['total']:.4f} "
            f"sdf={val_metrics.get('shape_sdf', 0.0):.4f}"
        )
        history.append({"epoch": epoch, "split": "val", **val_metrics})
        if wandb_run is not None:
            wandb_run.log({f"val/{key}": value for key, value in val_metrics.items()}, step=epoch)
        if float(val_metrics["total"]) < best_val:
            best_val = float(val_metrics["total"])
            save_checkpoint(output_dir / "best.pt", model, optimizer, scaler, config, epoch)

    history_path = save_json(output_dir / "metrics_history.json", {"history": history})
    print(f"saved metrics to {history_path}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
