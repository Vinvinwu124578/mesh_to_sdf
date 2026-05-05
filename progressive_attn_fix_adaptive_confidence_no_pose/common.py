from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
TACTISTRUCT_ROOT = Path(r"C:\Users\wudaw\Downloads\Tactistruct-main\Tactistruct-main")
TACTISTRUCT_SRC = TACTISTRUCT_ROOT / "src"
DEFAULT_SHAPENET_ROOT = Path(r"C:\Users\wudaw\Downloads\ShapeNetCore\ShapeNetCore")


def ensure_runtime_paths() -> None:
    for path in (WORKSPACE_ROOT, TACTISTRUCT_ROOT, TACTISTRUCT_SRC):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


ensure_runtime_paths()


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
    train_count, val_count, test_count = compute_split_counts(
        len(files),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
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


def resolve_split_files(split_cfg: dict) -> list[Path]:
    data_dir = (Path(split_cfg["root"]) / split_cfg.get("split", ".")).resolve()
    recursive = bool(split_cfg.get("recursive", False))
    all_files = sorted(data_dir.rglob("*.npz") if recursive else data_dir.glob("*.npz"))
    if not all_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    object_filenames = split_cfg.get("object_filenames")
    if object_filenames:
        relative_map = {path.relative_to(data_dir).as_posix(): path for path in all_files}
        basename_map: dict[str, list[Path]] = {}
        for path in all_files:
            basename_map.setdefault(path.name, []).append(path)

        selected_files = []
        for name in object_filenames:
            normalized = Path(name).as_posix().lstrip("./")
            if normalized in relative_map:
                selected_files.append(relative_map[normalized])
                continue
            basename = Path(normalized).name
            matches = basename_map.get(basename, [])
            if len(matches) == 1:
                selected_files.append(matches[0])
        if selected_files:
            return selected_files

    object_indices = split_cfg.get("object_indices")
    if object_indices:
        return [all_files[int(index)] for index in object_indices]

    return all_files


def save_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def install_amp_safe_structured_oracle_patch() -> None:
    if torch is None:
        return

    import tactistruct_structured_oracle.model as structured_oracle_model_module

    def masked_max_amp_safe(values: "torch.Tensor", mask: "torch.Tensor", dim: int) -> "torch.Tensor":
        bool_mask = mask.to(device=values.device, dtype=torch.bool)
        if torch.is_floating_point(values):
            fill_value = torch.finfo(values.dtype).min
        else:
            fill_value = torch.iinfo(values.dtype).min
        neg_inf = torch.full_like(values, fill_value)
        masked_values = torch.where(bool_mask, values, neg_inf)
        max_values = masked_values.max(dim=dim).values
        valid = bool_mask.any(dim=dim)
        return torch.where(valid, max_values, torch.zeros_like(max_values))

    structured_oracle_model_module.masked_max = masked_max_amp_safe
