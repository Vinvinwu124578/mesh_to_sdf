from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
TACTISTRUCT_ROOT = Path(r"C:\Users\wudaw\Downloads\Tactistruct-main\Tactistruct-main")
sys.path.insert(0, str(TACTISTRUCT_ROOT / "src"))

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from tactistruct.data.dataset import NPZTactileDataset, SyntheticTactileDataset
from tactistruct_progressive_attn_fix.dataset import AdaptiveTouchNPZDataset
from tactistruct_progressive_attn_fix.inference_utils import (
    infer_mesh,
    load_model_from_checkpoint,
    save_mesh_preview,
)
from tactistruct_progressive_attn_fix.losses import compute_progressive_attn_losses, eikonal_loss
from tactistruct_progressive_attn_fix.model import ProgressiveCompletionSystemAttnFix
from tactistruct.utils.checkpoint import load_project_checkpoint

DEFAULT_OUTPUT_DIR = "outputs/progressive_completion_attn_fix_surface_normal_constraint"


def _subsample_aligned_rows(
    points: np.ndarray,
    normals: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) != len(normals):
        raise ValueError(
            f"touch_points and touch_point_normals must have the same length, got {len(points)} and {len(normals)}."
        )
    if count <= 0:
        return points, normals
    if len(points) >= count:
        indices = rng.choice(len(points), size=count, replace=False)
        return points[indices], normals[indices]
    extra = rng.choice(len(points), size=count - len(points), replace=True)
    indices = np.concatenate([np.arange(len(points)), extra], axis=0)
    return points[indices], normals[indices]


def load_touch_points_from_file_optional_normals(
    touch_file: str | Path,
    touch_key: str = "touch_points",
    touch_view_index: int = 0,
    touch_group_key: str | None = None,
    touch_group_value: int | None = None,
    use_touch_point_normals: bool = False,
    touch_point_normals_key: str = "touch_point_normals",
) -> tuple[torch.Tensor, dict[str, int | str | bool | None]]:
    with np.load(touch_file) as payload:
        resolved_touch_key = touch_key
        if resolved_touch_key not in payload:
            if "touch_point_sets" in payload:
                resolved_touch_key = "touch_point_sets"
            else:
                raise KeyError(
                    f"Could not find tactile key '{touch_key}' in {touch_file}. Available keys: {list(payload.keys())}"
                )

        touch_points = np.asarray(payload[resolved_touch_key], dtype=np.float32)
        original_ndim = touch_points.ndim
        resolved_group_value = touch_group_value

        normals = None
        if use_touch_point_normals:
            if touch_point_normals_key not in payload:
                raise KeyError(
                    f"Could not find tactile normals key '{touch_point_normals_key}' in {touch_file}. "
                    f"Available keys: {list(payload.keys())}"
                )
            normals = np.asarray(payload[touch_point_normals_key], dtype=np.float32)
            if normals.shape != touch_points.shape:
                raise ValueError(
                    f"Normal key '{touch_point_normals_key}' is not aligned with tactile key '{resolved_touch_key}'. "
                    f"Expected shape {touch_points.shape}, got {normals.shape}."
                )

        if touch_points.ndim == 3:
            if touch_view_index < 0 or touch_view_index >= touch_points.shape[0]:
                raise IndexError(
                    f"touch_view_index {touch_view_index} is out of range for key '{resolved_touch_key}' "
                    f"with {touch_points.shape[0]} tactile views."
                )
            touch_points = touch_points[touch_view_index]
            if normals is not None:
                normals = normals[touch_view_index]
        elif touch_group_key is not None:
            if touch_group_key not in payload:
                raise KeyError(
                    f"Could not find tactile group key '{touch_group_key}' in {touch_file}. "
                    f"Available keys: {list(payload.keys())}"
                )
            group_values = payload[touch_group_key]
            unique_values = np.unique(group_values)
            if resolved_group_value is None:
                resolved_group_value = int(unique_values[0])
            mask = group_values == resolved_group_value
            if mask.ndim != 1 or mask.shape[0] != touch_points.shape[0]:
                raise ValueError(
                    f"Group key '{touch_group_key}' is not aligned with touch points. "
                    f"Expected length {touch_points.shape[0]}, got shape {group_values.shape}."
                )
            if not np.any(mask):
                raise ValueError(f"No touch points matched {touch_group_key}={resolved_group_value} in {touch_file}.")
            touch_points = touch_points[mask]
            if normals is not None:
                normals = normals[mask]

        if normals is not None:
            touch_points = np.concatenate([touch_points, normals], axis=-1).astype(np.float32)

        metadata = {
            "touch_key": resolved_touch_key,
            "touch_view_index": touch_view_index if original_ndim == 3 else None,
            "touch_group_key": touch_group_key,
            "touch_group_value": resolved_group_value,
            "num_touch_points": int(touch_points.shape[0]),
            "use_touch_point_normals": bool(use_touch_point_normals),
        }
        return torch.from_numpy(touch_points).unsqueeze(0), metadata


class AdaptiveTouchNPZDatasetOptionalNormals(AdaptiveTouchNPZDataset):
    def __init__(
        self,
        *args,
        use_touch_point_normals: bool = False,
        touch_point_normals_key: str = "touch_point_normals",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.use_touch_point_normals = bool(use_touch_point_normals)
        self.touch_point_normals_key = str(touch_point_normals_key)

    def _load_raw_touch_normals(
        self,
        payload: dict,
        view_type: str,
        view_value: int | None,
    ) -> np.ndarray | None:
        if not self.use_touch_point_normals:
            return None
        if self.touch_point_normals_key not in payload:
            raise KeyError(
                f"Could not find tactile normals key '{self.touch_point_normals_key}' in dataset payload. "
                f"Available keys: {list(payload.keys())}"
            )
        if view_type == "touch_sets":
            return np.asarray(payload[self.touch_point_normals_key][view_value], dtype=np.float32)
        if view_type == "touch_tensor":
            return np.asarray(payload[self.touch_point_normals_key][view_value], dtype=np.float32)
        if view_type == "touch_group":
            mask = payload[self.touch_group_key] == view_value
            return np.asarray(payload[self.touch_point_normals_key][mask], dtype=np.float32)
        if view_type == "touch_points":
            return np.asarray(payload[self.touch_point_normals_key], dtype=np.float32)
        if view_type == "touch_sets":
            return None
        return None

    def _sample_touch_points(
        self,
        payload: dict,
        selected_specs: list[tuple[str, int | None]],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int, int]:
        if not self.use_touch_point_normals:
            return super()._sample_touch_points(payload, selected_specs, rng)

        raw_point_chunks = []
        raw_normal_chunks = []
        for view_type, view_value in selected_specs:
            raw_touch_points = self._load_raw_touch_points(payload, view_type, view_value)
            raw_touch_normals = self._load_raw_touch_normals(payload, view_type, view_value)
            if raw_touch_points is None or len(raw_touch_points) <= 0:
                continue
            if raw_touch_normals is None:
                raise KeyError(
                    f"Selected tactile view requires '{self.touch_point_normals_key}', but that key is missing or "
                    "not supported for the current tactile layout."
                )
            if len(raw_touch_points) != len(raw_touch_normals):
                raise ValueError(
                    f"Tactile points and normals must align per view, got {len(raw_touch_points)} and "
                    f"{len(raw_touch_normals)}."
                )
            raw_point_chunks.append(raw_touch_points.astype(np.float32))
            raw_normal_chunks.append(raw_touch_normals.astype(np.float32))

        if not raw_point_chunks:
            raise ValueError("AdaptiveTouchNPZDatasetOptionalNormals could not load tactile points for the selected specs.")

        merged_touch_points = np.concatenate(raw_point_chunks, axis=0).astype(np.float32)
        merged_touch_normals = np.concatenate(raw_normal_chunks, axis=0).astype(np.float32)

        if self.min_touch_points >= self.num_touch_points:
            effective_count = int(self.num_touch_points)
        else:
            effective_count = int(rng.integers(self.min_touch_points, self.num_touch_points + 1))

        effective_touch_points, effective_touch_normals = _subsample_aligned_rows(
            merged_touch_points,
            merged_touch_normals,
            effective_count,
            rng,
        )
        fixed_touch_points, fixed_touch_normals = _subsample_aligned_rows(
            effective_touch_points,
            effective_touch_normals,
            self.num_touch_points,
            rng,
        )
        touch_features = np.concatenate([fixed_touch_points, fixed_touch_normals], axis=-1).astype(np.float32)
        return touch_features, effective_count, len(selected_specs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the fixed attention-pooled tactile completion variant on a multi-object NPZ dataset."
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing object .npz files.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--touch-key", type=str, default="touch_points")
    parser.add_argument("--use-touch-point-normals", action="store_true")
    parser.add_argument("--touch-point-normals-key", type=str, default="touch_point_normals")
    parser.add_argument(
        "--surface-normal-loss-weight",
        type=float,
        default=0.03,
        help="Weight for the surface normal consistency loss.",
    )
    parser.add_argument(
        "--surface-normal-loss-samples",
        type=int,
        default=512,
        help="How many surface points per batch to use for normal consistency.",
    )
    parser.add_argument(
        "--surface-normal-start-epoch",
        type=int,
        default=10,
        help="Keep surface normal loss disabled through this epoch, then enable it afterwards.",
    )
    parser.add_argument(
        "--surface-normal-ramp-epochs",
        type=int,
        default=5,
        help="Linearly ramp the surface normal loss weight over this many epochs after it starts.",
    )
    parser.add_argument(
        "--near-surface-ratio",
        type=float,
        default=0.4,
        help="Fraction of sampled query points biased toward the near-surface band.",
    )
    parser.add_argument(
        "--negative-sample-ratio",
        type=float,
        default=0.6,
        help="Fraction of sampled query points biased toward negative/inside examples.",
    )
    parser.add_argument(
        "--touch-zero-weight",
        type=float,
        default=0.25,
        help="Weight for the touch_zero loss term.",
    )
    parser.add_argument(
        "--surface-zero-weight",
        type=float,
        default=0.1,
        help="Weight for the surface_zero loss term.",
    )
    parser.add_argument("--num-touch-points", type=int, default=512)
    parser.add_argument("--min-touch-points", type=int, default=128)
    parser.add_argument("--merge-touch-views-prob", type=float, default=0.5)
    parser.add_argument("--max-merged-touch-views", type=int, default=4)
    parser.add_argument("--disable-adaptive-touch-augmentation", action="store_true")
    parser.add_argument("--cache-train-in-memory", action="store_true")
    parser.add_argument("--cache-val-in-memory", action="store_true")
    parser.add_argument("--train-touch-view-indices", type=str, default=None)
    parser.add_argument("--eval-touch-view-indices", type=str, default=None)
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
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="auto",
        choices=("auto", "fp16", "bf16"),
        help="AMP autocast dtype. 'auto' prefers bf16 when the GPU supports it.",
    )
    parser.add_argument(
        "--amp-warmup-epochs",
        type=int,
        default=3,
        help="Disable AMP for the first N epochs, then enable it afterwards.",
    )
    parser.add_argument(
        "--disable-amp-safe-loss-fp32",
        action="store_true",
        help="Keep loss computation inside autocast instead of forcing it back to fp32.",
    )
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--enable-wandb", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <output-dir>/latest.pt and continue training from the next epoch.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from a specific checkpoint path instead of <output-dir>/latest.pt.",
    )
    parser.add_argument(
        "--resume-next-epoch",
        type=int,
        default=0,
        help="Override the next epoch when resuming from an older checkpoint without saved epoch metadata.",
    )
    return parser.parse_args()


def _cast_float_tensors_to_fp32(value):
    if torch.is_tensor(value):
        if torch.is_floating_point(value):
            return value.float()
        return value
    if isinstance(value, dict):
        return {key: _cast_float_tensors_to_fp32(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cast_float_tensors_to_fp32(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cast_float_tensors_to_fp32(item) for item in value)
    return value


def resolve_amp_dtype(amp_cfg: dict, device: torch.device) -> torch.dtype:
    requested = str(amp_cfg.get("dtype", "auto")).strip().lower()
    bf16_supported = bool(
        device.type == "cuda"
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    if requested == "bf16":
        if bf16_supported:
            return torch.bfloat16
        print("amp dtype override | bf16 requested but not supported on this GPU, falling back to fp16")
        return torch.float16
    if requested == "fp16":
        return torch.float16
    return torch.bfloat16 if bf16_supported else torch.float16


def format_amp_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    return str(dtype)


def subsample_aligned_tensors(
    points: torch.Tensor,
    normals: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if count <= 0 or points.size(1) <= count:
        return points, normals
    indices = torch.randperm(points.size(1), device=points.device)[:count]
    return points[:, indices], normals[:, indices]


def compute_surface_normal_consistency_loss(
    model: ProgressiveCompletionSystemAttnFix,
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    sample_count: int,
    train_mode: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if "surface_points" not in batch or "surface_normals" not in batch:
        return None, None
    if "patch_tokens" not in outputs or "latent_path" not in outputs:
        return None, None

    surface_points = batch["surface_points"].float()
    surface_normals = batch["surface_normals"].float()
    if surface_points.ndim != 3 or surface_normals.shape != surface_points.shape:
        return None, None

    surface_points, surface_normals = subsample_aligned_tensors(surface_points, surface_normals, int(sample_count))
    surface_query_points = surface_points.detach().clone().requires_grad_(True)
    target_normals = torch.nn.functional.normalize(surface_normals.detach(), dim=-1, eps=1e-6)

    touch_points = batch.get("touch_points")
    touch_xyz = None if touch_points is None else touch_points[..., :3].float()
    touch_count_ratio = batch.get("touch_count_ratio")
    if touch_count_ratio is not None:
        touch_count_ratio = touch_count_ratio.float()

    with torch.enable_grad():
        stage_sdf_fp32, _ = model.decode_points(
            surface_query_points,
            outputs["patch_tokens"].float(),
            outputs["latent_path"].float(),
            touch_points=touch_xyz,
            touch_count_ratio=touch_count_ratio,
        )
        predicted_sdf = stage_sdf_fp32[:, -1].unsqueeze(-1)
        gradients = torch.autograd.grad(
            outputs=predicted_sdf.sum(),
            inputs=surface_query_points,
            create_graph=train_mode,
            retain_graph=train_mode,
            only_inputs=True,
        )[0]

    gradients = torch.nn.functional.normalize(gradients.float(), dim=-1, eps=1e-6)
    cosine = (gradients * target_normals).sum(dim=-1).clamp(-1.0, 1.0)
    loss = (1.0 - cosine).mean()
    mean_dot = cosine.mean()
    return loss, mean_dot


def resolve_surface_normal_weight(loss_cfg: dict, epoch: int) -> float:
    base_weight = float(loss_cfg.get("surface_normal", 0.0))
    if base_weight <= 0.0:
        return 0.0
    start_epoch = int(loss_cfg.get("surface_normal_start_epoch", 0))
    ramp_epochs = int(loss_cfg.get("surface_normal_ramp_epochs", 0))
    if int(epoch) <= start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return base_weight
    progress = min(1.0, max(0.0, float(int(epoch) - start_epoch) / float(ramp_epochs)))
    return base_weight * progress


def maybe_init_wandb(config: dict, output_dir: Path):
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb logging is enabled in the config, but the 'wandb' package is not installed. "
            "Please run `pip install wandb` or disable wandb."
        ) from exc

    return wandb.init(
        project=wandb_cfg.get("project", "tactistruct-progressive-attn-fix"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name"),
        group=wandb_cfg.get("group"),
        tags=wandb_cfg.get("tags"),
        notes=wandb_cfg.get("notes"),
        mode=wandb_cfg.get("mode", "online"),
        dir=str(output_dir),
        config=config,
    )


def parse_index_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",")]
    items = [item for item in items if item]
    if not items:
        return []
    return [int(item) for item in items]


def collect_npz_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    return files


def _reduce_splits_for_small_datasets(num_items: int, ratios: np.ndarray) -> np.ndarray:
    ratios = ratios.copy()
    while int(np.count_nonzero(ratios > 0.0)) > num_items:
        if ratios[1] > 0.0:
            ratios[1] = 0.0
            continue
        if ratios[2] > 0.0:
            ratios[2] = 0.0
            continue
        break
    return ratios


def compute_split_counts(
    num_items: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios < 0.0):
        raise ValueError("Split ratios must be non-negative.")
    if ratios.sum() <= 0.0:
        raise ValueError("At least one split ratio must be greater than zero.")

    ratios = _reduce_splits_for_small_datasets(num_items, ratios)
    normalized = ratios / ratios.sum()
    raw_counts = normalized * num_items
    counts = np.floor(raw_counts).astype(int)
    remainder = int(num_items - counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw_counts - counts))
        for split_index in order[:remainder]:
            counts[split_index] += 1

    for split_index, ratio in enumerate(ratios):
        if ratio > 0.0 and counts[split_index] == 0:
            donor_indices = np.argsort(-counts)
            donor = next((idx for idx in donor_indices if counts[idx] > 1), None)
            if donor is None:
                break
            counts[donor] -= 1
            counts[split_index] += 1

    train_count, val_count, test_count = (int(value) for value in counts.tolist())
    if train_count <= 0:
        raise ValueError("Training split must contain at least one object.")
    return train_count, val_count, test_count


def split_object_files(
    files: list[Path],
    data_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    train_count, val_count, test_count = compute_split_counts(len(files), train_ratio, val_ratio, test_ratio)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(files))
    shuffled = [files[index] for index in permutation.tolist()]
    train_files = shuffled[:train_count]
    val_files = shuffled[train_count : train_count + val_count]
    test_files = shuffled[train_count + val_count : train_count + val_count + test_count]
    return (
        [path.relative_to(data_dir).as_posix() for path in train_files],
        [path.relative_to(data_dir).as_posix() for path in val_files],
        [path.relative_to(data_dir).as_posix() for path in test_files],
    )


def build_dataset_cfg(
    data_dir: Path,
    object_filenames: list[str],
    touch_key: str,
    use_touch_point_normals: bool,
    touch_point_normals_key: str,
    touch_view_indices: list[int] | None,
    seed: int,
    num_touch_points: int,
    adaptive_touch_cfg: dict | None = None,
    cache_in_memory: bool = False,
) -> dict:
    cfg = {
        "name": "npz",
        "root": str(data_dir.resolve()),
        "split": ".",
        "object_filenames": object_filenames,
        "recursive": True,
        "touch_views_key": touch_key,
        "num_surface_points": 4096,
        "num_touch_points": int(num_touch_points),
        "num_query_points": 6144,
        "dynamic_sampling": True,
        "query_sampling_mode": "precomputed",
        "near_surface_ratio": 0.6,
        "near_surface_threshold": 0.015,
        "negative_sample_ratio": 0.6,
        "bbox_padding": 0.2,
        "reference_surface_points": 16384,
        "sdf_clip": 0.2,
        "seed": seed,
        "cache_in_memory": bool(cache_in_memory),
        "use_touch_point_normals": bool(use_touch_point_normals),
        "touch_point_normals_key": str(touch_point_normals_key),
    }
    if touch_view_indices is not None:
        cfg["touch_view_indices"] = touch_view_indices
    if adaptive_touch_cfg is not None:
        cfg["adaptive_touch"] = adaptive_touch_cfg
    return cfg


def build_config(args: argparse.Namespace) -> tuple[dict, dict[str, list[str]]]:
    data_dir = Path(args.data_dir).resolve()
    files = collect_npz_files(data_dir)
    if int(args.num_touch_points) <= 0:
        raise ValueError("--num-touch-points must be greater than zero.")
    if int(args.min_touch_points) <= 0:
        raise ValueError("--min-touch-points must be greater than zero.")
    if int(args.max_merged_touch_views) <= 0:
        raise ValueError("--max-merged-touch-views must be greater than zero.")
    if int(args.prefetch_factor) <= 0:
        raise ValueError("--prefetch-factor must be greater than zero.")
    if int(args.val_every) <= 0:
        raise ValueError("--val-every must be greater than zero.")
    if not 0.0 <= float(args.merge_touch_views_prob) <= 1.0:
        raise ValueError("--merge-touch-views-prob must be in [0, 1].")
    if not 0.0 <= float(args.near_surface_ratio) <= 1.0:
        raise ValueError("--near-surface-ratio must be in [0, 1].")
    if not 0.0 <= float(args.negative_sample_ratio) <= 1.0:
        raise ValueError("--negative-sample-ratio must be in [0, 1].")
    if int(args.surface_normal_start_epoch) < 0:
        raise ValueError("--surface-normal-start-epoch must be >= 0.")
    if int(args.surface_normal_ramp_epochs) < 0:
        raise ValueError("--surface-normal-ramp-epochs must be >= 0.")

    train_touch_indices = parse_index_list(args.train_touch_view_indices)
    eval_touch_indices = parse_index_list(args.eval_touch_view_indices)
    train_files, val_files, test_files = split_object_files(
        files=files,
        data_dir=data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir).resolve()
    run_name = output_dir.name
    adaptive_touch_cfg = None
    if not args.disable_adaptive_touch_augmentation:
        adaptive_touch_cfg = {
            "enabled": True,
            "min_touch_points": min(int(args.min_touch_points), int(args.num_touch_points)),
            "merge_touch_views_prob": float(args.merge_touch_views_prob),
            "max_merged_views": int(args.max_merged_touch_views),
        }
    config = {
        "seed": int(args.seed),
        "dataset": {
            "train": build_dataset_cfg(
                data_dir,
                train_files,
                args.touch_key,
                bool(args.use_touch_point_normals),
                args.touch_point_normals_key,
                train_touch_indices,
                args.seed,
                num_touch_points=args.num_touch_points,
                adaptive_touch_cfg=adaptive_touch_cfg,
                cache_in_memory=bool(args.cache_train_in_memory),
            ),
        },
        "model": {
            "use_touch_conditioning": True,
            "encoder": {
                "input_dim": 6 if bool(args.use_touch_point_normals) else 3,
                "hidden_dim": 256,
                "token_dim": 256,
                "num_patches": 24,
                "points_per_patch": 48,
                "position_bands": 8,
                "global_attention_heads": 4,
                "noise_std": 0.01,
                "rotation_degrees": 10.0,
            },
            "latent_path": {
                "latent_dim": 256,
                "hidden_dim": 512,
                "num_stages": 6,
                "max_noise_std": 0.4,
                "min_noise_std": 0.05,
            },
            "decoder": {
                "hidden_dim": 512,
                "num_heads": 8,
                "position_bands": 8,
                "touch_position_bands": 6,
                "touch_topk": 16,
                "touch_sigma": 0.08,
            },
            "active_sampling": {
                "enabled": True,
                "topk_points": 2048,
            },
            "surface_query_samples": 2048,
        },
        "loss": {
            "sdf": 1.0,
            "progressive_sdf": 0.75,
            "stage_weights": [0.15, 0.3, 0.5, 0.7, 0.85, 1.0],
            "active_focus": 0.5,
            "touch_zero": float(args.touch_zero_weight),
            "surface": float(args.surface_zero_weight),
            "eikonal": 0,
            "patch": 0.25,
            "attention_entropy": 0.01,
            "attention_diversity": 0.05,
            "path_smoothness": 0.05,
            "surface_normal": float(args.surface_normal_loss_weight),
            "surface_normal_samples": int(args.surface_normal_loss_samples),
            "surface_normal_start_epoch": int(args.surface_normal_start_epoch),
            "surface_normal_ramp_epochs": int(args.surface_normal_ramp_epochs),
        },
        "train": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "prefetch_factor": int(args.prefetch_factor),
            "val_every": int(args.val_every),
            "amp": {
                "enabled": not bool(args.disable_amp),
                "dtype": str(args.amp_dtype),
                "warmup_epochs": int(args.amp_warmup_epochs),
                "safe_loss_fp32": not bool(args.disable_amp_safe_loss_fp32),
            },
            "progress": {"enabled": not bool(args.disable_progress)},
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "device": str(args.device),
            "output_dir": str(output_dir),
        },
        "wandb": {
            "enabled": bool(args.enable_wandb),
            "project": "tactistruct-progressive-attn-fix",
            "name": run_name,
            "mode": "online",
            "tags": [
                "progressive-attn-fix",
                "touch-aware-decoder",
                "adaptive-touch",
                "shape-net",
                "multi-object",
                "surface-normal-constraint",
                "touch-point-normals" if bool(args.use_touch_point_normals) else "touch-xyz-only",
            ],
        },
        "post_train_infer": {
            "enabled": False,
            "checkpoint": "latest",
            "touch_file": str((data_dir / Path(test_files[0])).resolve()) if test_files else None,
            "touch_key": args.touch_key,
            "use_touch_point_normals": bool(args.use_touch_point_normals),
            "touch_point_normals_key": args.touch_point_normals_key,
            "touch_view_index": int(eval_touch_indices[0]) if eval_touch_indices else 0,
            "resolution": 128,
            "chunk_size": 32768,
            "touch_subsample_mode": "random",
            "conditioning_touch_point_count": int(args.num_touch_points),
            "decoder_touch_point_count": int(args.num_touch_points),
            "output": "post_train_infer/first_test_object.ply",
            "preview_image": "post_train_infer/first_test_object.png",
        },
    }
    if val_files:
        config["dataset"]["val"] = build_dataset_cfg(
            data_dir,
            val_files,
            args.touch_key,
            bool(args.use_touch_point_normals),
            args.touch_point_normals_key,
            eval_touch_indices,
            args.seed + 1000,
            num_touch_points=args.num_touch_points,
            cache_in_memory=bool(args.cache_val_in_memory),
        )
    if test_files:
        config["dataset"]["test"] = build_dataset_cfg(
            data_dir,
            test_files,
            args.touch_key,
            bool(args.use_touch_point_normals),
            args.touch_point_normals_key,
            eval_touch_indices,
            args.seed + 2000,
            num_touch_points=args.num_touch_points,
            cache_in_memory=False,
        )
    for split_name in tuple(config["dataset"].keys()):
        config["dataset"][split_name]["near_surface_ratio"] = float(args.near_surface_ratio)
        config["dataset"][split_name]["negative_sample_ratio"] = float(args.negative_sample_ratio)

    split_summary = {"train": train_files, "val": val_files, "test": test_files}
    return config, split_summary


def save_generated_config(config: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "progressive_attn_fix_generated_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path


def print_split_summary(split_summary: dict[str, list[str]]) -> None:
    print("Progressive completion attn-fix split summary")
    for split_name in ("train", "val", "test"):
        filenames = split_summary.get(split_name, [])
        preview = ", ".join(filenames[:5])
        if len(filenames) > 5:
            preview += ", ..."
        print(f"  {split_name}: {len(filenames)} objects")
        if preview:
            print(f"    {preview}")


def extract_split_summary_from_config(config: dict) -> dict[str, list[str]]:
    summary: dict[str, list[str]] = {}
    dataset_cfg = config.get("dataset", {})
    for split_name in ("train", "val", "test"):
        split_cfg = dataset_cfg.get(split_name, {})
        summary[split_name] = list(split_cfg.get("object_filenames", []))
    return summary


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(cfg: dict) -> torch.utils.data.Dataset:
    name = cfg["name"].lower()
    params = {key: value for key, value in cfg.items() if key != "name"}
    if name == "npz":
        adaptive_touch_cfg = params.pop("adaptive_touch", None)
        use_touch_point_normals = bool(params.pop("use_touch_point_normals", False))
        touch_point_normals_key = str(params.pop("touch_point_normals_key", "touch_point_normals"))
        if isinstance(adaptive_touch_cfg, dict) and adaptive_touch_cfg.get("enabled", False):
            adaptive_params = {key: value for key, value in adaptive_touch_cfg.items() if key != "enabled"}
            if use_touch_point_normals:
                return AdaptiveTouchNPZDatasetOptionalNormals(
                    **params,
                    use_touch_point_normals=True,
                    touch_point_normals_key=touch_point_normals_key,
                    **adaptive_params,
                )
            return AdaptiveTouchNPZDataset(**params, **adaptive_params)
        if use_touch_point_normals:
            return AdaptiveTouchNPZDatasetOptionalNormals(
                **params,
                use_touch_point_normals=True,
                touch_point_normals_key=touch_point_normals_key,
            )
        return NPZTactileDataset(**params)
    if name == "synthetic":
        return SyntheticTactileDataset(**params)
    raise ValueError(f"Unsupported dataset type: {cfg['name']}")


def build_dataloaders(
    config: dict,
    train_dataset: torch.utils.data.Dataset | None = None,
    val_dataset: torch.utils.data.Dataset | None = None,
) -> tuple[DataLoader, DataLoader | None]:
    train_cfg = config["dataset"]["train"]
    if train_dataset is None:
        train_dataset = build_dataset(train_cfg)
    batch_size = int(config["train"]["batch_size"])
    configured_num_workers = int(config["train"].get("num_workers", 0))
    device_name = str(config["train"].get("device", "cpu")).lower()
    use_pin_memory = device_name == "cuda"

    def make_loader_kwargs(dataset: torch.utils.data.Dataset) -> dict:
        dataset_num_workers = configured_num_workers
        if sys.platform.startswith("win") and bool(getattr(dataset, "cache_in_memory", False)) and dataset_num_workers > 0:
            print("data loader | cache_in_memory enabled on Windows, forcing num_workers=0 to avoid duplicating RAM")
            dataset_num_workers = 0
        kwargs = {
            "batch_size": batch_size,
            "num_workers": dataset_num_workers,
            "drop_last": False,
            "pin_memory": use_pin_memory,
        }
        if dataset_num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = max(2, int(config["train"].get("prefetch_factor", 2)))
        return kwargs

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **make_loader_kwargs(train_dataset),
    )
    val_loader = None
    if "val" in config["dataset"]:
        if val_dataset is None:
            val_dataset = build_dataset(config["dataset"]["val"])
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **make_loader_kwargs(val_dataset),
        )
    return train_loader, val_loader


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved = {}
    non_blocking = device.type == "cuda"
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=non_blocking) if torch.is_tensor(value) else value
    return moved


def summarise_metrics(metrics: dict[str, float], batches: int) -> dict[str, float]:
    return {key: value / max(batches, 1) for key, value in metrics.items()}


def sanitize_loss_dict(losses: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    sanitized: dict[str, torch.Tensor] = {}
    nonfinite_terms = 0
    for key, value in losses.items():
        if torch.is_tensor(value):
            if torch.isfinite(value).all():
                sanitized[key] = value
            else:
                sanitized[key] = torch.zeros((), device=device, dtype=torch.float32)
                nonfinite_terms += 1
        else:
            sanitized[key] = value
    if nonfinite_terms > 0:
        sanitized["nonfinite_terms"] = torch.tensor(float(nonfinite_terms), device=device)
    return sanitized


def collect_nonfinite_gradients(model: torch.nn.Module, limit: int | None = None) -> list[str]:
    names: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if torch.isfinite(parameter.grad).all():
            continue
        names.append(name)
        if limit is not None and len(names) >= limit:
            break
    return names


def print_dataset_cache_summary(split_name: str, dataset: torch.utils.data.Dataset | None) -> None:
    if dataset is None or not bool(getattr(dataset, "cache_in_memory", False)):
        return
    cached_bytes = int(getattr(dataset, "cached_bytes", 0))
    cached_gb = cached_bytes / float(1024**3)
    print(f"dataset cache | {split_name} preloaded into RAM ({cached_gb:.2f} GB)")


def capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object] | None) -> None:
    if not state:
        return
    python_state = state.get("python")
    numpy_state = state.get("numpy")
    torch_state = state.get("torch")
    cuda_state = state.get("cuda")
    if python_state is not None:
        random.setstate(python_state)
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def build_checkpoint_payload(
    model: ProgressiveCompletionSystemAttnFix,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    config: dict,
    epoch: int,
) -> dict[str, object]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "config": config,
        "epoch": int(epoch),
        "train_state": {
            "rng_state": capture_rng_state(),
        },
    }


def resolve_resume_checkpoint(args: argparse.Namespace, output_dir: Path) -> Path | None:
    if args.resume_from:
        return Path(args.resume_from).expanduser().resolve()
    if args.resume:
        return (output_dir / "latest.pt").resolve()
    return None


def load_resume_checkpoint(resume_path: Path) -> dict[str, object]:
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    checkpoint = load_project_checkpoint(resume_path, map_location="cpu")
    if "model" not in checkpoint:
        raise KeyError(f"Resume checkpoint {resume_path} does not contain a model state.")
    return checkpoint


def merge_resume_config(
    args: argparse.Namespace,
    checkpoint_config: dict,
    fallback_config: dict,
    resume_path: Path,
) -> dict:
    config = json.loads(json.dumps(checkpoint_config or fallback_config))

    if args.resume_from and args.output_dir == DEFAULT_OUTPUT_DIR:
        config["train"]["output_dir"] = str(resume_path.parent.resolve())
    else:
        config["train"]["output_dir"] = str(Path(args.output_dir).resolve())

    config["train"]["epochs"] = int(args.epochs)
    config["train"]["batch_size"] = int(args.batch_size)
    config["train"]["num_workers"] = int(args.num_workers)
    config["train"]["prefetch_factor"] = int(args.prefetch_factor)
    config["train"]["val_every"] = int(args.val_every)
    config["train"]["amp"] = {
        "enabled": not bool(args.disable_amp),
        "dtype": str(args.amp_dtype),
        "warmup_epochs": int(args.amp_warmup_epochs),
        "safe_loss_fp32": not bool(args.disable_amp_safe_loss_fp32),
    }
    config["train"]["progress"] = {"enabled": not bool(args.disable_progress)}
    config["train"]["lr"] = float(args.lr)
    config["train"]["weight_decay"] = float(args.weight_decay)
    config["train"]["grad_clip"] = float(args.grad_clip)
    config["train"]["device"] = str(args.device)
    if args.cache_train_in_memory:
        config["dataset"]["train"]["cache_in_memory"] = True
    if args.cache_val_in_memory and "val" in config.get("dataset", {}):
        config["dataset"]["val"]["cache_in_memory"] = True
    for split_name in ("train", "val", "test"):
        if split_name in config.get("dataset", {}):
            config["dataset"][split_name]["near_surface_ratio"] = float(args.near_surface_ratio)
            config["dataset"][split_name]["negative_sample_ratio"] = float(args.negative_sample_ratio)
    use_touch_point_normals = bool(args.use_touch_point_normals)
    for split_name in ("train", "val", "test"):
        if split_name in config.get("dataset", {}):
            config["dataset"][split_name]["use_touch_point_normals"] = use_touch_point_normals
            config["dataset"][split_name]["touch_point_normals_key"] = str(args.touch_point_normals_key)
    config.setdefault("model", {}).setdefault("encoder", {})
    config["model"]["encoder"]["input_dim"] = 6 if use_touch_point_normals else 3
    config.setdefault("post_train_infer", {})
    config["post_train_infer"]["use_touch_point_normals"] = use_touch_point_normals
    config["post_train_infer"]["touch_point_normals_key"] = str(args.touch_point_normals_key)
    config.setdefault("loss", {})
    config["loss"]["touch_zero"] = float(args.touch_zero_weight)
    config["loss"]["surface"] = float(args.surface_zero_weight)
    config["loss"]["surface_normal"] = float(args.surface_normal_loss_weight)
    config["loss"]["surface_normal_samples"] = int(args.surface_normal_loss_samples)
    config["loss"]["surface_normal_start_epoch"] = int(args.surface_normal_start_epoch)
    config["loss"]["surface_normal_ramp_epochs"] = int(args.surface_normal_ramp_epochs)
    config["wandb"]["enabled"] = bool(args.enable_wandb)
    config["wandb"]["name"] = Path(config["train"]["output_dir"]).resolve().name
    return config


def restore_training_state(
    checkpoint: dict[str, object],
    model: ProgressiveCompletionSystemAttnFix,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None = None,
    resume_next_epoch_override: int = 0,
) -> tuple[int, int]:
    model.load_state_dict(checkpoint["model"])

    train_state = checkpoint.get("train_state", {})
    if not isinstance(train_state, dict):
        train_state = {}

    optimizer_state = checkpoint.get("optimizer")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    scaler_state = checkpoint.get("scaler")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    restore_rng_state(train_state.get("rng_state"))

    completed_epoch = int(checkpoint.get("epoch", 0) or 0)
    if resume_next_epoch_override > 0:
        next_epoch = int(resume_next_epoch_override)
    elif completed_epoch > 0:
        next_epoch = completed_epoch + 1
    else:
        next_epoch = 1

    return completed_epoch, next_epoch


def run_epoch(
    model: ProgressiveCompletionSystemAttnFix,
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
    amp_dtype: torch.dtype,
    amp_safe_loss_fp32: bool,
    amp_warmup_epochs: int,
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

    eikonal_weight = float(loss_cfg.get("eikonal", 0.0))
    surface_normal_weight = resolve_surface_normal_weight(loss_cfg, epoch=epoch)
    surface_normal_samples = int(loss_cfg.get("surface_normal_samples", 0))
    primary_loss_cfg = dict(loss_cfg)
    primary_loss_cfg["eikonal"] = 0.0
    primary_loss_cfg["surface_normal"] = 0.0
    primary_loss_cfg["surface_normal_samples"] = surface_normal_samples

    for batch_index, batch in enumerate(iterator, start=1):
        batch = move_to_device(batch, device)
        if train_mode and float(loss_cfg.get("eikonal", 0.0)) > 0.0 and "query_points" in batch:
            batch["query_points"] = batch["query_points"].clone().detach().requires_grad_(True)

        with torch.set_grad_enabled(train_mode):
            amp_active = bool(
                amp_enabled
                and device.type == "cuda"
                and int(epoch) > int(max(0, amp_warmup_epochs))
            )
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if amp_active
                else nullcontext()
            )
            with autocast_context:
                outputs = model(batch, apply_augmentation=train_mode)

            if amp_safe_loss_fp32:
                batch_for_loss = _cast_float_tensors_to_fp32(batch)
                outputs_for_loss = _cast_float_tensors_to_fp32(outputs)
            else:
                batch_for_loss = batch
                outputs_for_loss = outputs

            with torch.autocast(device_type="cuda", enabled=False):
                losses = compute_progressive_attn_losses(
                    batch_for_loss,
                    outputs_for_loss,
                    primary_loss_cfg,
                    include_eikonal=False,
                )
                total = losses["total"]

            if surface_normal_weight > 0.0:
                surface_normal_value, surface_normal_dot = compute_surface_normal_consistency_loss(
                    model=model,
                    batch=batch,
                    outputs=outputs,
                    sample_count=surface_normal_samples,
                    train_mode=train_mode,
                )
                if surface_normal_value is not None and torch.isfinite(surface_normal_value):
                    losses["surface_normal"] = surface_normal_value
                    total = total + surface_normal_weight * surface_normal_value
                else:
                    losses["surface_normal"] = torch.zeros((), device=device)
                    losses["surface_normal_skipped"] = torch.ones((), device=device)
                if surface_normal_dot is not None and torch.isfinite(surface_normal_dot):
                    losses["surface_normal_mean_dot"] = surface_normal_dot
            if surface_normal_weight > 0.0:
                losses["surface_normal_weight_applied"] = torch.tensor(
                    float(surface_normal_weight),
                    device=device,
                    dtype=torch.float32,
                )

            if train_mode and eikonal_weight > 0.0 and "query_points" in batch and "sdf" in outputs:
                eikonal_query_points = batch["query_points"].detach().clone().float().requires_grad_(True)
                eikonal_touch_points = None
                if batch.get("touch_points") is not None:
                    eikonal_touch_points = batch["touch_points"][..., :3].float()
                eikonal_touch_count_ratio = batch.get("touch_count_ratio")
                if eikonal_touch_count_ratio is not None:
                    eikonal_touch_count_ratio = eikonal_touch_count_ratio.float()
                stage_sdf_fp32, _ = model.decode_points(
                    eikonal_query_points,
                    outputs["patch_tokens"].float(),
                    outputs["latent_path"].float(),
                    touch_points=eikonal_touch_points,
                    touch_count_ratio=eikonal_touch_count_ratio,
                )
                eikonal_value = eikonal_loss(eikonal_query_points, stage_sdf_fp32[:, -1].unsqueeze(-1))

                if torch.isfinite(eikonal_value):
                    losses["eikonal"] = eikonal_value
                    total = total + eikonal_weight * eikonal_value
                else:
                    losses["eikonal"] = torch.zeros((), device=device)
                    losses["eikonal_skipped"] = torch.ones((), device=device)

            if not torch.isfinite(total):
                losses["nonfinite_total"] = torch.ones((), device=device)
                total = torch.zeros((), device=device)
            losses["total"] = total

            if train_mode and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                if float(losses.get("nonfinite_total", torch.zeros((), device=device)).item()) <= 0.0:
                    if scaler is not None and scaler.is_enabled() and amp_active and device.type == "cuda":
                        scaler.scale(total).backward()
                        nonfinite_gradients = collect_nonfinite_gradients(model, limit=16)
                        if nonfinite_gradients:
                            optimizer.zero_grad(set_to_none=True)
                            losses["nonfinite_grad"] = torch.ones((), device=device)
                            losses["nonfinite_grad_count"] = torch.tensor(
                                float(len(nonfinite_gradients)),
                                device=device,
                            )
                        else:
                            if grad_clip is not None and grad_clip > 0:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                            scaler.step(optimizer)
                            scaler.update()
                    else:
                        total.backward()
                        nonfinite_gradients = collect_nonfinite_gradients(model, limit=16)
                        if nonfinite_gradients:
                            optimizer.zero_grad(set_to_none=True)
                            losses["nonfinite_grad"] = torch.ones((), device=device)
                            losses["nonfinite_grad_count"] = torch.tensor(
                                float(len(nonfinite_gradients)),
                                device=device,
                            )
                        else:
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


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    config: dict,
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_checkpoint_payload(model, optimizer, scaler, config, epoch), path)


def run_post_training_inference(
    config: dict,
    output_dir: Path,
    device: torch.device,
    wandb_run=None,
) -> None:
    infer_cfg = config.get("post_train_infer", {})
    if not infer_cfg.get("enabled", False):
        return

    checkpoint_name = infer_cfg.get("checkpoint", "latest").lower()
    checkpoint_path = output_dir / f"{checkpoint_name}.pt"
    if not checkpoint_path.exists():
        checkpoint_path = output_dir / "latest.pt"
    model, _ = load_model_from_checkpoint(checkpoint_path, config=config, device=device)

    touch_points, touch_metadata = load_touch_points_from_file_optional_normals(
        touch_file=infer_cfg["touch_file"],
        touch_key=infer_cfg.get("touch_key", "touch_points"),
        touch_view_index=int(infer_cfg.get("touch_view_index", 0)),
        use_touch_point_normals=bool(infer_cfg.get("use_touch_point_normals", False)),
        touch_point_normals_key=str(infer_cfg.get("touch_point_normals_key", "touch_point_normals")),
    )

    mesh_path = output_dir / infer_cfg.get("output", "post_train_infer/reconstruction.ply")
    preview_path = output_dir / infer_cfg.get("preview_image", "post_train_infer/reconstruction.png")
    vertices, faces, stats = infer_mesh(
        model=model,
        touch_points=touch_points,
        resolution=int(infer_cfg.get("resolution", 64)),
        chunk_size=int(infer_cfg.get("chunk_size", 32768)),
        conditioning_touch_point_count=int(infer_cfg.get("conditioning_touch_point_count", 0)) or None,
        decoder_touch_point_count=int(infer_cfg.get("decoder_touch_point_count", 0)) or None,
        touch_subsample_mode=str(infer_cfg.get("touch_subsample_mode", "random")),
        device=device,
    )
    saved_preview = save_mesh_preview(vertices, faces, mesh_path, preview_path)
    print(
        f"post-train infer | sdf min={stats['sdf_min']:.6f} max={stats['sdf_max']:.6f} "
        f"mean={stats['sdf_mean']:.6f} | vertices={int(stats['num_vertices'])} "
        f"faces={int(stats['num_faces'])} | touch_points={touch_metadata['num_touch_points']}"
    )
    if saved_preview is not None:
        print(f"post-train infer | preview={saved_preview}")

    if wandb_run is not None:
        wandb_run.log(
            {
                "post_train/sdf_min": stats["sdf_min"],
                "post_train/sdf_max": stats["sdf_max"],
                "post_train/sdf_mean": stats["sdf_mean"],
                "post_train/num_vertices": stats["num_vertices"],
                "post_train/num_faces": stats["num_faces"],
            }
        )


def main() -> None:
    args = parse_args()
    fallback_config, fallback_split_summary = build_config(args)
    resume_path = resolve_resume_checkpoint(args, Path(fallback_config["train"]["output_dir"]))
    resume_checkpoint = None
    config = fallback_config
    split_summary = fallback_split_summary

    if resume_path is not None:
        resume_checkpoint = load_resume_checkpoint(resume_path)
        checkpoint_config = resume_checkpoint.get("config")
        if isinstance(checkpoint_config, dict):
            config = merge_resume_config(args, checkpoint_config, fallback_config, resume_path)
            split_summary = extract_split_summary_from_config(config)
        print(f"resume training | checkpoint={resume_path}")

    output_dir = Path(config["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print_split_summary(split_summary)

    set_seed(int(config.get("seed", 42)))
    torch.set_float32_matmul_precision("high")
    device_name = config["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_dataset = build_dataset(config["dataset"]["train"])
    val_dataset = build_dataset(config["dataset"]["val"]) if "val" in config["dataset"] else None
    print_dataset_cache_summary("train", train_dataset)
    print_dataset_cache_summary("val", val_dataset)

    model_cfg = config["model"]
    model = ProgressiveCompletionSystemAttnFix(
        encoder_cfg=model_cfg.get("encoder", {}),
        latent_path_cfg=model_cfg.get("latent_path", {}),
        decoder_cfg=model_cfg.get("decoder", {}),
        active_sampling_cfg=model_cfg.get("active_sampling", {}),
        surface_query_samples=int(model_cfg.get("surface_query_samples", 512)),
        use_touch_conditioning=bool(model_cfg.get("use_touch_conditioning", True)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"].get("lr", 1e-3)),
        weight_decay=float(config["train"].get("weight_decay", 1e-6)),
    )
    amp_cfg = dict(config["train"].get("amp", {}))
    amp_enabled = bool(amp_cfg.get("enabled", True)) and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(amp_cfg, device)
    amp_warmup_epochs = int(amp_cfg.get("warmup_epochs", 0))
    amp_safe_loss_fp32 = bool(amp_cfg.get("safe_loss_fp32", True))
    if amp_enabled and sys.platform.startswith("win") and float(config.get("loss", {}).get("eikonal", 0.0)) > 0.0:
        print("amp override | disabled AMP on Windows CUDA because eikonal loss uses higher-order gradients and can crash")
        amp_enabled = False
        amp_cfg["enabled"] = False
        config["train"]["amp"] = amp_cfg
    show_progress = bool(config["train"].get("progress", {}).get("enabled", True))
    scaler_enabled = bool(amp_enabled and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    if amp_enabled:
        print(
            "amp config | "
            f"dtype={format_amp_dtype(amp_dtype)} "
            f"warmup_epochs={amp_warmup_epochs} "
            f"loss_fp32={amp_safe_loss_fp32} "
            f"grad_scaler={scaler_enabled}"
        )
    else:
        print("amp config | disabled")

    grad_clip = config["train"].get("grad_clip", 1.0)
    loss_cfg = config.get("loss", {})

    config_path = save_generated_config(config, output_dir)
    print(f"saved generated config to {config_path}")
    shutil.copyfile(config_path, output_dir / "config.json")
    wandb_run = maybe_init_wandb(config, output_dir)

    train_loader, val_loader = build_dataloaders(config, train_dataset=train_dataset, val_dataset=val_dataset)
    epochs = int(config["train"].get("epochs", 20))
    best_val = float("inf")
    completed_epoch = 0
    start_epoch = 1
    if resume_checkpoint is not None:
        completed_epoch, start_epoch = restore_training_state(
            resume_checkpoint,
            model,
            optimizer,
            scaler=scaler,
            resume_next_epoch_override=args.resume_next_epoch,
        )
        if "optimizer" not in resume_checkpoint:
            print("resume warning | checkpoint has no optimizer state; continuing with a fresh optimizer")
        if "epoch" not in resume_checkpoint and args.resume_next_epoch <= 0:
            print(
                "resume warning | checkpoint has no saved epoch metadata; restarting from epoch 1. "
                "Use --resume-next-epoch <n> for older checkpoints if needed."
            )
        print(f"resume state | completed_epoch={completed_epoch} next_epoch={start_epoch}")

    if start_epoch > epochs:
        print(
            f"resume training | checkpoint already reached epoch {completed_epoch}, "
            f"which is >= requested total epochs {epochs}. Increase --epochs to continue."
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            loss_cfg,
            device,
            grad_clip,
            train_mode=True,
            epoch=epoch,
            total_epochs=epochs,
            show_progress=show_progress,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            amp_safe_loss_fp32=amp_safe_loss_fp32,
            amp_warmup_epochs=amp_warmup_epochs,
        )
        line = f"epoch {epoch:03d} | train total={train_metrics.get('total', 0.0):.4f}"
        extras = [f"{key}={value:.4f}" for key, value in train_metrics.items() if key != "total"]
        if extras:
            line += " | " + ", ".join(extras)

        log_payload = {"epoch": epoch}
        log_payload.update({f"train/{key}": value for key, value in train_metrics.items()})

        run_validation = val_loader is not None and (
            epoch % max(1, int(config["train"].get("val_every", 1))) == 0 or epoch == epochs
        )
        if run_validation:
            val_metrics = run_epoch(
                model,
                val_loader,
                None,
                None,
                loss_cfg,
                device,
                grad_clip,
                train_mode=False,
                epoch=epoch,
                total_epochs=epochs,
                show_progress=show_progress,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                amp_safe_loss_fp32=amp_safe_loss_fp32,
                amp_warmup_epochs=amp_warmup_epochs,
            )
            val_total = val_metrics.get("total", 0.0)
            line += f" | val total={val_total:.4f}"
            extras = [f"val_{key}={value:.4f}" for key, value in val_metrics.items() if key != "total"]
            if extras:
                line += " | " + ", ".join(extras)
            log_payload.update({f"val/{key}": value for key, value in val_metrics.items()})
            if val_total < best_val:
                best_val = val_total
                save_checkpoint(output_dir / "best.pt", model, optimizer, scaler, config, epoch)

        save_checkpoint(output_dir / "latest.pt", model, optimizer, scaler, config, epoch)
        if wandb_run is not None:
            wandb_run.log(log_payload, step=epoch)
        print(line)

    run_post_training_inference(config=config, output_dir=output_dir, device=device, wandb_run=wandb_run)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
