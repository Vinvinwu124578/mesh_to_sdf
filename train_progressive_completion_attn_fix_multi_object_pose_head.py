from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


TACTISTRUCT_ROOT = Path(r"C:\Users\wudaw\Downloads\Tactistruct-main\Tactistruct-main")
DEFAULT_DATA_DIR = Path(
    r"C:\Users\wudaw\OneDrive - University of Bristol\Desktop"
    r"\3D_Printing_Objects\watertight_tactile_pose_dataset_32"
)
DEFAULT_OUTPUT_DIR = TACTISTRUCT_ROOT / "outputs" / "progressive_attn_fix_pose_head_3dprint_32_small_data"
POSE_SUFFIX_RE = re.compile(r"__pose_\d+$")


def _pop_arg_value(argv: list[str], name: str, default: str | None = None) -> str | None:
    prefix = name + "="
    for index, value in enumerate(list(argv)):
        if value.startswith(prefix):
            argv.pop(index)
            return value[len(prefix) :]
        if value == name:
            argv.pop(index)
            if index >= len(argv):
                raise ValueError(f"{name} requires a value.")
            return argv.pop(index)
    return default


def _pop_flag(argv: list[str], name: str) -> bool:
    removed = False
    while name in argv:
        argv.remove(name)
        removed = True
    return removed


def _has_arg(argv: list[str], name: str) -> bool:
    return any(value == name or value.startswith(name + "=") for value in argv)


def _append_default(argv: list[str], name: str, value: str | int | float | Path) -> None:
    if not _has_arg(argv, name):
        argv.extend([name, str(value)])


def _pose_source_id(path: Path) -> str:
    return POSE_SUFFIX_RE.sub("", path.stem)


def _split_pose_files_by_source_object(
    files: list[Path],
    data_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    import train_progressive_completion_attn_fix_multi_object_small_data as base_train

    groups: dict[str, list[Path]] = {}
    for path in files:
        groups.setdefault(_pose_source_id(path), []).append(path)

    group_keys = sorted(groups)
    train_count, val_count, test_count = base_train.compute_split_counts(
        len(group_keys),
        train_ratio,
        val_ratio,
        test_ratio,
    )
    rng = np.random.default_rng(seed)
    shuffled = [group_keys[index] for index in rng.permutation(len(group_keys)).tolist()]
    train_keys = set(shuffled[:train_count])
    val_keys = set(shuffled[train_count : train_count + val_count])
    test_keys = set(shuffled[train_count + val_count : train_count + val_count + test_count])

    def flatten(keys: set[str]) -> list[str]:
        selected: list[Path] = []
        for key in sorted(keys):
            selected.extend(sorted(groups[key]))
        return [path.relative_to(data_dir).as_posix() for path in selected]

    return flatten(train_keys), flatten(val_keys), flatten(test_keys)


class ProgressiveCompletionSystemAttnFixPoseHeadAdapter:
    """Factory wrapper matching the constructor used by the original train script."""

    def __new__(
        cls,
        encoder_cfg: dict | None = None,
        latent_path_cfg: dict | None = None,
        decoder_cfg: dict | None = None,
        active_sampling_cfg: dict | None = None,
        surface_query_samples: int = 512,
        use_touch_conditioning: bool = True,
        decode_touch_sdf: bool = True,
        decode_surface_sdf: bool = True,
    ):
        from tactistruct_progressive_attn_fix_pose import ProgressiveCompletionSystemAttnFixPose

        class ProgressiveCompletionSystemAttnFixPoseHead(ProgressiveCompletionSystemAttnFixPose):
            def forward(
                self,
                batch,
                apply_augmentation: bool = False,
                use_oracle_rotation: bool = False,
            ):
                force_oracle = bool(getattr(self, "force_oracle_rotation", False))
                return super().forward(
                    batch,
                    apply_augmentation=apply_augmentation,
                    use_oracle_rotation=bool(use_oracle_rotation or force_oracle),
                )

        del decode_touch_sdf, decode_surface_sdf
        return ProgressiveCompletionSystemAttnFixPoseHead(
            encoder_cfg=encoder_cfg,
            latent_path_cfg=latent_path_cfg,
            decoder_cfg=decoder_cfg,
            active_sampling_cfg=active_sampling_cfg,
            pose_head_cfg={"hidden_dim": 256},
            surface_query_samples=surface_query_samples,
            use_touch_conditioning=use_touch_conditioning,
        )


def main() -> None:
    sys.path.insert(0, str(TACTISTRUCT_ROOT))
    sys.path.insert(0, str(TACTISTRUCT_ROOT / "src"))

    forwarded_argv = [sys.argv[0], *sys.argv[1:]]
    rotation_loss_weight = float(_pop_arg_value(forwarded_argv, "--rotation-loss-weight", "0.5"))
    rotation_teacher_forcing_epochs = int(
        _pop_arg_value(forwarded_argv, "--rotation-teacher-forcing-epochs", "30")
    )
    group_pose_splits = not _pop_flag(forwarded_argv, "--file-level-split")

    _append_default(forwarded_argv, "--data-dir", DEFAULT_DATA_DIR)
    _append_default(forwarded_argv, "--output-dir", DEFAULT_OUTPUT_DIR)
    _append_default(forwarded_argv, "--device", "cuda")
    import train_progressive_completion_attn_fix_multi_object_small_data as base_train
    from tactistruct_progressive_attn_fix_pose import (
        PoseAdaptiveTouchNPZDataset,
        compute_progressive_attn_pose_losses,
    )

    original_build_config = base_train.build_config

    def build_config_with_pose(args: argparse.Namespace):
        config, split_summary = original_build_config(args)
        config.setdefault("loss", {})["rotation"] = float(rotation_loss_weight)
        config.setdefault("model", {})["pose_head"] = {"hidden_dim": 256}
        config.setdefault("post_train_infer", {})["enabled"] = False
        config.setdefault("pose_head_training", {})
        config["pose_head_training"].update(
            {
                "enabled": True,
                "rotation_loss_weight": float(rotation_loss_weight),
                "rotation_teacher_forcing_epochs": int(rotation_teacher_forcing_epochs),
                "group_pose_splits_by_source_object": bool(group_pose_splits),
                "base_train_script": str(TACTISTRUCT_ROOT / "train_progressive_completion_attn_fix_multi_object_small_data.py"),
            }
        )
        config.setdefault("train", {})["rotation_teacher_forcing_epochs"] = int(rotation_teacher_forcing_epochs)
        config.setdefault("wandb", {}).setdefault("tags", [])
        for tag in ("pose-head", "posed-tactile", "3d-print"):
            if tag not in config["wandb"]["tags"]:
                config["wandb"]["tags"].append(tag)
        return config, split_summary

    original_run_epoch = base_train.run_epoch

    def run_epoch_with_pose_teacher_forcing(
        model,
        loader,
        optimizer,
        scaler,
        loss_cfg,
        device,
        grad_clip,
        train_mode,
        epoch,
        total_epochs,
        show_progress,
        amp_enabled,
    ):
        force_oracle = bool(train_mode and int(epoch) <= int(rotation_teacher_forcing_epochs))
        previous_force_oracle = getattr(model, "force_oracle_rotation", False)
        setattr(model, "force_oracle_rotation", force_oracle)
        try:
            metrics = original_run_epoch(
                model,
                loader,
                optimizer,
                scaler,
                loss_cfg,
                device,
                grad_clip,
                train_mode,
                epoch,
                total_epochs,
                show_progress,
                amp_enabled,
            )
        finally:
            setattr(model, "force_oracle_rotation", previous_force_oracle)
        metrics["rotation_teacher_forced"] = float(force_oracle)
        return metrics

    base_train.AdaptiveTouchNPZDataset = PoseAdaptiveTouchNPZDataset
    base_train.ProgressiveCompletionSystemAttnFix = ProgressiveCompletionSystemAttnFixPoseHeadAdapter
    base_train.compute_progressive_attn_losses = compute_progressive_attn_pose_losses
    base_train.build_config = build_config_with_pose
    base_train.run_epoch = run_epoch_with_pose_teacher_forcing
    if group_pose_splits:
        base_train.split_object_files = _split_pose_files_by_source_object

    sys.argv = forwarded_argv
    base_train.main()


if __name__ == "__main__":
    main()
