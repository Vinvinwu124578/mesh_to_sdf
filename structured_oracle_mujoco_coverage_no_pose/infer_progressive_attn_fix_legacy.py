from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from common import collect_npz_files, load_json, parse_index_list, resolve_split_files, save_json
from dataset_legacy_progressive import load_structured_touch_points_from_file
from tactistruct_progressive_attn_fix.inference_utils import (
    infer_mesh,
    load_model_from_checkpoint,
    save_mesh_preview,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference with the legacy progressive_attn_fix-style model on structured MuJoCo "
            "coverage-aware data without modifying the original progressive_attn_fix package."
        )
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--touch-view-indices", type=str, default=None)
    parser.add_argument("--merge-touch-views", action="store_true")
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=32768)
    parser.add_argument("--conditioning-touch-point-count", type=int, default=0)
    parser.add_argument("--decoder-touch-point-count", type=int, default=0)
    parser.add_argument("--touch-subsample-mode", type=str, default="random", choices=["random", "fps"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def resolve_input_files(args: argparse.Namespace, config: dict) -> list[Path]:
    if args.input_path is not None:
        input_path = Path(args.input_path).resolve()
        if input_path.is_file():
            return [input_path]
        if input_path.is_dir():
            return collect_npz_files(input_path)
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    dataset_cfg = config["dataset"][args.split]
    return resolve_split_files(dataset_cfg)


def list_view_indices(touch_file: Path, touch_view_indices: list[int] | None) -> list[int]:
    with np.load(touch_file, mmap_mode="r") as payload:
        if "patch_points" not in payload:
            raise KeyError(f"Could not find 'patch_points' in {touch_file}.")
        available_count = int(payload["patch_points"].shape[0])
    if touch_view_indices is None:
        return list(range(available_count))
    invalid_indices = [index for index in touch_view_indices if index < 0 or index >= available_count]
    if invalid_indices:
        raise IndexError(
            f"touch_view_indices contains out-of-range values {invalid_indices}; "
            f"file only has {available_count} tactile views."
        )
    return [int(index) for index in touch_view_indices]


def build_object_output_id(object_path: Path, object_index: int) -> str:
    stem = object_path.stem
    category = stem.split("__")[0] if "__" in stem else stem.split("_")[0]
    category = "".join(ch for ch in category if ch.isalnum()).lower() or "object"
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:10]
    return f"obj_{object_index:04d}_{category}_{digest}"


def _resolve_touch_point_count(requested_count: int, config: dict) -> int | None:
    if requested_count > 0:
        return int(requested_count)
    dataset_cfg = config.get("dataset", {}).get("train", {})
    default_count = int(dataset_cfg.get("num_touch_points", 512))
    return default_count if default_count > 0 else None


def main() -> None:
    args = parse_args()
    config = None if args.config is None else load_json(args.config)
    device = torch.device(args.device)
    model, resolved_config = load_model_from_checkpoint(args.checkpoint, config=config, device=device)

    files = resolve_input_files(args, resolved_config)
    if args.max_objects > 0:
        files = files[: int(args.max_objects)]

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else Path(args.checkpoint).resolve().parent / "inference_progressive_attn_fix_legacy"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    touch_view_indices = parse_index_list(args.touch_view_indices)
    conditioning_touch_point_count = _resolve_touch_point_count(args.conditioning_touch_point_count, resolved_config)
    decoder_touch_point_count = _resolve_touch_point_count(args.decoder_touch_point_count, resolved_config)
    rows = []
    failures = []

    for object_index, object_path in enumerate(files, start=1):
        try:
            object_output_dir = output_dir / build_object_output_id(object_path, object_index)
            object_output_dir.mkdir(parents=True, exist_ok=True)

            with np.load(object_path) as payload:
                mesh_name = (
                    str(payload["mesh_name"].item())
                    if "mesh_name" in payload and np.asarray(payload["mesh_name"]).shape == ()
                    else object_path.stem
                )
                coverage = (
                    float(payload["planning_surface_coverage_ratio"])
                    if "planning_surface_coverage_ratio" in payload
                    else None
                )

            resolved_views = list_view_indices(object_path, touch_view_indices)
            touch_specs = [("merged", resolved_views)] if args.merge_touch_views else [
                (f"view{int(view_index):02d}", [int(view_index)]) for view_index in resolved_views
            ]

            for label, selected_views in touch_specs:
                touch_points, metadata = load_structured_touch_points_from_file(
                    touch_file=object_path,
                    touch_view_indices=selected_views,
                )
                vertices, faces, stats = infer_mesh(
                    model=model,
                    touch_points=touch_points,
                    resolution=int(args.resolution),
                    chunk_size=int(args.chunk_size),
                    conditioning_touch_point_count=conditioning_touch_point_count,
                    decoder_touch_point_count=decoder_touch_point_count,
                    touch_subsample_mode=str(args.touch_subsample_mode),
                    device=device,
                    return_intermediates=False,
                )
                mesh_path = object_output_dir / f"{label}.ply"
                preview_path = object_output_dir / f"{label}.png" if bool(args.save_preview) else None
                save_mesh_preview(vertices, faces, mesh_path, preview_path)
                row = {
                    "object_path": str(object_path),
                    "mesh_name": mesh_name,
                    "touch_label": label,
                    "touch_view_indices": selected_views,
                    "mesh_path": str(mesh_path),
                    "planning_surface_coverage_ratio": coverage,
                    "num_touch_points": int(metadata["num_touch_points"]),
                    **stats,
                }
                rows.append(row)
                print(
                    f"[OK] {object_path.name} touch={label} "
                    f"verts={int(stats['num_vertices'])} faces={int(stats['num_faces'])}"
                )
        except Exception as exc:
            failures.append({"object_path": str(object_path), "error": str(exc)})
            print(f"[FAILED] {object_path}")
            print(exc)
            if bool(args.fail_fast):
                raise

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_objects": len(files),
        "num_meshes": len(rows),
        "failures": failures,
        "rows": rows,
    }
    summary_path = save_json(output_dir / "inference_summary.json", summary)
    print(f"[SUMMARY] saved to {summary_path}")


if __name__ == "__main__":
    main()
