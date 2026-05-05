from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from common import (
    collect_npz_files,
    install_amp_safe_structured_oracle_patch,
    load_json,
    parse_index_list,
    resolve_split_files,
    save_json,
)
from tactistruct.utils.checkpoint import load_project_checkpoint
from tactistruct.utils.geometry import create_query_grid, extract_mesh_from_sdf, write_ply
from tactistruct_structured_oracle.model import StructuredTouchOracleSystem


install_amp_safe_structured_oracle_patch()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference for the no-pose structured oracle model on coverage-aware MuJoCo structured data."
        )
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--touch-view-indices", type=str, default=None)
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=32768)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    config: dict | None = None,
    device: torch.device | None = None,
) -> tuple[StructuredTouchOracleSystem, dict]:
    checkpoint = load_project_checkpoint(checkpoint_path, map_location="cpu")
    resolved_config = checkpoint.get("config") if config is None else config
    if resolved_config is None:
        raise ValueError("Could not resolve config for inference.")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = resolved_config["model"]
    model = StructuredTouchOracleSystem(
        patch_encoder_cfg=model_cfg.get("patch_encoder", {}),
        round_fusion_cfg=model_cfg.get("round_fusion", {}),
        global_encoder_cfg=model_cfg.get("global_encoder", {}),
        decoder_cfg=model_cfg.get("decoder", {}),
        surface_query_samples=int(model_cfg.get("surface_query_samples", 2048)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, resolved_config


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


def selected_view_indices(payload, touch_view_indices: list[int] | None) -> list[int]:
    patch_points = payload["patch_points"]
    if patch_points.ndim != 5:
        raise ValueError(f"Expected patch_points with shape [V, F, R, P, 3], got {patch_points.shape}.")
    available_count = int(patch_points.shape[0])
    if touch_view_indices is None:
        return list(range(available_count))
    invalid_indices = [index for index in touch_view_indices if index < 0 or index >= available_count]
    if invalid_indices:
        raise IndexError(
            f"touch_view_indices contains out-of-range values {invalid_indices}; "
            f"file only has {available_count} tactile views."
        )
    return [int(index) for index in touch_view_indices]


def build_batch_for_view(payload, view_index: int) -> dict[str, torch.Tensor]:
    patch_points = torch.from_numpy(np.asarray(payload["patch_points"][view_index], dtype=np.float32)).unsqueeze(0)
    patch_centers = torch.from_numpy(np.asarray(payload["patch_centers"][view_index], dtype=np.float32)).unsqueeze(0)
    patch_radii = torch.from_numpy(np.asarray(payload["patch_radii"][view_index], dtype=np.float32)).unsqueeze(0)
    patch_mask = torch.from_numpy(np.asarray(payload["patch_mask"][view_index], dtype=np.bool_)).unsqueeze(0)
    finger_mask = torch.from_numpy(np.asarray(payload["finger_mask"][view_index], dtype=np.bool_)).unsqueeze(0)
    pose_rotation = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    pose_translation = torch.zeros((1, 3), dtype=torch.float32)
    return {
        "patch_points": patch_points,
        "patch_centers": patch_centers,
        "patch_radii": patch_radii,
        "patch_mask": patch_mask,
        "finger_mask": finger_mask,
        "pose_rotation": pose_rotation,
        "pose_translation": pose_translation,
    }


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved = {}
    non_blocking = device.type == "cuda"
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=non_blocking)
    return moved


def infer_mesh(
    model: StructuredTouchOracleSystem,
    batch: dict[str, torch.Tensor],
    resolution: int,
    chunk_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    batch = move_to_device(batch, device)
    with torch.no_grad():
        aligned_patch_points, aligned_patch_centers = model.canonicalizer(
            patch_points=batch["patch_points"],
            patch_centers=batch["patch_centers"],
            pose_rotation=batch["pose_rotation"],
            pose_translation=batch["pose_translation"],
        )
        conditioning = model.encode_structure(
            patch_points=aligned_patch_points,
            patch_centers=aligned_patch_centers,
            patch_radii=batch["patch_radii"],
            patch_mask=batch["patch_mask"],
            finger_mask=batch["finger_mask"],
        )
        grid = create_query_grid(resolution).to(device)
        sdf_chunks = []
        for start in range(0, grid.size(0), chunk_size):
            chunk = grid[start : start + chunk_size].unsqueeze(0)
            sdf, _ = model.decode_points(
                chunk,
                conditioning["shape_latent"],
                conditioning["finger_tokens"],
                aligned_patch_centers,
                batch["finger_mask"],
            )
            sdf_chunks.append(sdf.squeeze(0).squeeze(-1).cpu())

    sdf_values = torch.cat(sdf_chunks, dim=0)
    stats = {
        "sdf_min": float(sdf_values.min().item()),
        "sdf_max": float(sdf_values.max().item()),
        "sdf_mean": float(sdf_values.mean().item()),
    }
    if not (stats["sdf_min"] <= 0.0 <= stats["sdf_max"]):
        raise RuntimeError(
            "Predicted SDF does not cross zero on the query grid. "
            f"Observed range: [{stats['sdf_min']:.6f}, {stats['sdf_max']:.6f}]"
        )
    vertices, faces = extract_mesh_from_sdf(sdf_values, resolution)
    stats["num_vertices"] = float(len(vertices))
    stats["num_faces"] = float(len(faces))
    return vertices, faces, stats


def build_object_output_id(object_path: Path, object_index: int) -> str:
    stem = object_path.stem
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:10]
    return f"obj_{object_index:04d}_{digest}"


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
        else Path(args.checkpoint).resolve().parent / "inference_structured_oracle_mujoco_coverage_no_pose"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    touch_view_indices = parse_index_list(args.touch_view_indices)
    rows = []
    failures = []

    for object_index, object_path in enumerate(files):
        try:
            with np.load(object_path) as payload:
                view_indices = selected_view_indices(payload, touch_view_indices)
                object_output_dir = output_dir / build_object_output_id(object_path, object_index)
                object_output_dir.mkdir(parents=True, exist_ok=True)

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

                for view_index in view_indices:
                    batch = build_batch_for_view(payload, view_index)
                    vertices, faces, stats = infer_mesh(
                        model=model,
                        batch=batch,
                        resolution=int(args.resolution),
                        chunk_size=int(args.chunk_size),
                        device=device,
                    )
                    mesh_path = object_output_dir / f"view_{int(view_index):02d}.ply"
                    write_ply(mesh_path, vertices, faces)
                    row = {
                        "object_path": str(object_path),
                        "mesh_name": mesh_name,
                        "view_index": int(view_index),
                        "mesh_path": str(mesh_path),
                        "planning_surface_coverage_ratio": coverage,
                        **stats,
                    }
                    rows.append(row)
                    print(
                        f"[OK] {object_path.name} view={int(view_index):02d} "
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
