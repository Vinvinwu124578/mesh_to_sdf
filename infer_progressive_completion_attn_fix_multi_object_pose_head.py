from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


TACTISTRUCT_ROOT = Path(r"C:\Users\wudaw\Downloads\Tactistruct-main\Tactistruct-main")
DEFAULT_DATA_DIR = Path(
    r"C:\Users\wudaw\OneDrive - University of Bristol\Desktop"
    r"\3D_Printing_Objects\watertight_tactile_pose_dataset_32"
)
DEFAULT_CHECKPOINT = TACTISTRUCT_ROOT / "outputs" / "progressive_attn_fix_pose_head_3dprint_32" / "best.pt"

sys.path.insert(0, str(TACTISTRUCT_ROOT))
sys.path.insert(0, str(TACTISTRUCT_ROOT / "src"))

from tactistruct.inference_utils import save_inference_intermediates, save_mesh_preview
from tactistruct.utils.checkpoint import load_project_checkpoint
from tactistruct.utils.geometry import (
    create_query_grid,
    decimate_mesh_vertex_clustering,
    extract_mesh_from_sdf,
    smooth_mesh_laplacian,
    smooth_mesh_taubin,
)
from tactistruct_progressive_attn_fix.inference_utils import (
    compute_touch_count_ratio,
    list_touch_inputs_from_file,
    load_touch_points_from_file,
    subsample_touch_points,
)
from tactistruct_progressive_attn_fix_pose import (
    ProgressiveCompletionSystemAttnFixPose,
)
from tactistruct_progressive_attn_fix_pose.model import canonicalize_xyz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pose-head inference for progressive_completion_attn_fix multi-object checkpoints."
    )
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test", "all"))
    parser.add_argument("--touch-key", type=str, default=None)
    parser.add_argument("--touch-view-indices", type=str, default=None)
    parser.add_argument("--merge-touch-views", action="store_true", default=True)
    parser.add_argument("--single-touch-views", dest="merge_touch_views", action="store_false")
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=32768)
    parser.add_argument("--conditioning-touch-point-count", type=int, default=512)
    parser.add_argument("--decoder-touch-point-count", type=int, default=512)
    parser.add_argument("--touch-subsample-mode", type=str, default="fps", choices=("random", "fps"))
    parser.add_argument("--oracle-pose", action="store_true", help="Use pose_rotation from NPZ for debugging/evaluation.")
    parser.add_argument("--sdf-smooth-sigma", type=float, default=0.0)
    parser.add_argument("--mesh-decimate-voxel-size", type=float, default=0.0)
    parser.add_argument("--mesh-smooth-iterations", type=int, default=0)
    parser.add_argument("--mesh-smooth-method", type=str, default="taubin", choices=("taubin", "laplacian"))
    parser.add_argument("--mesh-smooth-lambda", type=float, default=0.25)
    parser.add_argument("--mesh-smooth-mu", type=float, default=-0.27)
    parser.add_argument("--binary-ply", action="store_true")
    parser.add_argument("--save-intermediates", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_index_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",")]
    items = [item for item in items if item]
    if not items:
        return []
    return [int(item) for item in items]


def load_pose_model(
    checkpoint_path: str | Path,
    config: dict | None,
    device: torch.device,
) -> tuple[ProgressiveCompletionSystemAttnFixPose, dict]:
    checkpoint = load_project_checkpoint(checkpoint_path, map_location="cpu")
    resolved_config = checkpoint.get("config") if config is None else config
    if resolved_config is None:
        raise ValueError("Could not resolve config from checkpoint or --config.")

    model_cfg = resolved_config["model"]
    model = ProgressiveCompletionSystemAttnFixPose(
        encoder_cfg=model_cfg.get("encoder", {}),
        latent_path_cfg=model_cfg.get("latent_path", {}),
        decoder_cfg=model_cfg.get("decoder", {}),
        active_sampling_cfg=model_cfg.get("active_sampling", {}),
        pose_head_cfg=model_cfg.get("pose_head", {"hidden_dim": 256}),
        surface_query_samples=int(model_cfg.get("surface_query_samples", 512)),
        use_touch_conditioning=bool(model_cfg.get("use_touch_conditioning", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, resolved_config


def resolve_split_files(config: dict, split: str, fallback_data_dir: Path) -> list[Path]:
    if split == "all":
        files = sorted(fallback_data_dir.rglob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files found under {fallback_data_dir}")
        return files

    split_cfg = config.get("dataset", {}).get(split)
    if split_cfg is None:
        files = sorted(fallback_data_dir.rglob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files found under {fallback_data_dir}")
        return files

    data_dir = (Path(split_cfg["root"]) / split_cfg.get("split", ".")).resolve()
    all_files = sorted(data_dir.rglob("*.npz") if split_cfg.get("recursive", False) else data_dir.glob("*.npz"))
    if not all_files:
        raise FileNotFoundError(f"No .npz files found under {data_dir}")

    names = split_cfg.get("object_filenames") or []
    if not names:
        return all_files

    relative_map = {path.relative_to(data_dir).as_posix(): path for path in all_files}
    basename_map: dict[str, list[Path]] = {}
    for path in all_files:
        basename_map.setdefault(path.name, []).append(path)

    selected: list[Path] = []
    missing: list[str] = []
    for name in names:
        normalized = Path(name).as_posix().lstrip("./")
        if normalized in relative_map:
            selected.append(relative_map[normalized])
            continue
        matches = basename_map.get(Path(normalized).name, [])
        if len(matches) == 1:
            selected.append(matches[0])
        else:
            missing.append(name)
    if missing and not selected:
        raise FileNotFoundError(f"Could not resolve split files: {missing[:10]}")
    return selected


def load_merged_touch_points_from_file(
    touch_file: str | Path,
    touch_key: str,
    touch_view_indices: list[int] | None,
) -> tuple[torch.Tensor, dict[str, int | str | None]]:
    with np.load(touch_file) as payload:
        if touch_key not in payload:
            raise KeyError(f"Could not find tactile key {touch_key!r} in {touch_file}.")
        touch_points = np.asarray(payload[touch_key], dtype=np.float32)
        if touch_points.ndim != 3:
            return torch.from_numpy(touch_points).unsqueeze(0), {
                "touch_key": touch_key,
                "touch_view_index": None,
                "touch_label": "touch",
                "num_touch_points": int(touch_points.shape[0]),
            }
        selected = list(range(touch_points.shape[0])) if touch_view_indices is None else [int(i) for i in touch_view_indices]
        invalid = [i for i in selected if i < 0 or i >= touch_points.shape[0]]
        if invalid:
            raise IndexError(f"touch_view_indices out of range for {touch_file}: {invalid}")
        merged = touch_points[selected].reshape(-1, touch_points.shape[-1])
        return torch.from_numpy(merged).unsqueeze(0), {
            "touch_key": touch_key,
            "touch_view_index": None,
            "touch_label": "merged",
            "num_touch_points": int(merged.shape[0]),
        }


def load_oracle_pose_rotation(path: Path, device: torch.device) -> torch.Tensor | None:
    with np.load(path) as payload:
        if "pose_rotation" not in payload:
            return None
        rotation = np.asarray(payload["pose_rotation"], dtype=np.float32)
    if rotation.shape != (3, 3):
        raise ValueError(f"pose_rotation must have shape (3, 3), got {rotation.shape} in {path}")
    return torch.from_numpy(rotation).unsqueeze(0).to(device)


def infer_pose_mesh(
    model: ProgressiveCompletionSystemAttnFixPose,
    touch_points: torch.Tensor,
    resolution: int,
    chunk_size: int,
    conditioning_touch_point_count: int,
    decoder_touch_point_count: int,
    touch_subsample_mode: str,
    sdf_smooth_sigma: float,
    mesh_decimate_voxel_size: float,
    mesh_smooth_method: str,
    mesh_smooth_iterations: int,
    mesh_smooth_lambda: float,
    mesh_smooth_mu: float,
    oracle_rotation: torch.Tensor | None = None,
    return_intermediates: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, torch.Tensor] | None]
):
    device = next(model.parameters()).device
    touch_points = touch_points.to(device)
    if touch_points.size(0) != 1:
        raise ValueError("Pose-head inference expects one object at a time.")

    conditioning_touch_points = subsample_touch_points(
        touch_points,
        conditioning_touch_point_count,
        mode=touch_subsample_mode,
    )
    decoder_touch_points = subsample_touch_points(
        touch_points,
        decoder_touch_point_count,
        mode=touch_subsample_mode,
    )
    touch_count_ratio = compute_touch_count_ratio(
        conditioning_touch_points=conditioning_touch_points,
        decoder_touch_points=decoder_touch_points,
        conditioning_touch_point_count=conditioning_touch_point_count,
        decoder_touch_point_count=decoder_touch_point_count,
    )
    if touch_count_ratio is not None:
        touch_count_ratio = touch_count_ratio.to(device)

    grid_world = create_query_grid(resolution).to(device)
    sdf_chunks = []
    stage_sdf_chunks: list[list[torch.Tensor]] | None = None

    with torch.no_grad():
        conditioning = model.encode_touch(
            batch_size=1,
            device=device,
            touch_points=conditioning_touch_points,
            touch_count_ratio=touch_count_ratio,
            use_oracle_rotation=oracle_rotation is not None,
            oracle_rotation=oracle_rotation,
        )
        canonical_touch_xyz = None
        if conditioning.get("canonical_touch_points") is not None:
            canonical_touch_xyz = conditioning["canonical_touch_points"][..., :3]
        if return_intermediates:
            stage_sdf_chunks = [[] for _ in range(conditioning["latent_path"].size(1))]

        canonicalization_rotation = conditioning["canonicalization_rotation"]
        for start in range(0, grid_world.size(0), chunk_size):
            world_chunk = grid_world[start : start + chunk_size].unsqueeze(0)
            canonical_chunk = canonicalize_xyz(world_chunk, canonicalization_rotation)
            stage_sdf, _ = model.decode_points(
                canonical_chunk,
                conditioning["patch_tokens"],
                conditioning["latent_path"],
                touch_points=canonical_touch_xyz,
                touch_count_ratio=touch_count_ratio,
            )
            sdf_chunks.append(stage_sdf[:, -1].squeeze(0).cpu())
            if stage_sdf_chunks is not None:
                for stage_index in range(stage_sdf.size(1)):
                    stage_sdf_chunks[stage_index].append(stage_sdf[:, stage_index].squeeze(0).cpu())

    sdf_values = torch.cat(sdf_chunks, dim=0)
    stats = {
        "sdf_min": float(sdf_values.min().item()),
        "sdf_max": float(sdf_values.max().item()),
        "sdf_mean": float(sdf_values.mean().item()),
        "touch_count_ratio": 0.0 if touch_count_ratio is None else float(touch_count_ratio.item()),
        "used_oracle_pose": float(oracle_rotation is not None),
    }
    pred_rotation = conditioning["pred_pose_rotation"].detach().cpu().numpy()[0]
    used_rotation = conditioning["canonicalization_rotation"].detach().cpu().numpy()[0]
    stats["pred_pose_trace"] = float(np.trace(pred_rotation))
    if not (stats["sdf_min"] <= 0.0 <= stats["sdf_max"]):
        raise RuntimeError(
            "Predicted SDF does not cross zero on the posed query grid. "
            f"Observed range: [{stats['sdf_min']:.6f}, {stats['sdf_max']:.6f}]"
        )

    vertices, faces = extract_mesh_from_sdf(sdf_values, resolution, smooth_sigma=float(sdf_smooth_sigma))
    vertices, faces = decimate_mesh_vertex_clustering(vertices, faces, voxel_size=float(mesh_decimate_voxel_size))
    if int(mesh_smooth_iterations) > 0:
        if mesh_smooth_method == "taubin":
            vertices = smooth_mesh_taubin(
                vertices,
                faces,
                iterations=int(mesh_smooth_iterations),
                lambd=float(mesh_smooth_lambda),
                mu=float(mesh_smooth_mu),
            )
        else:
            vertices = smooth_mesh_laplacian(
                vertices,
                faces,
                iterations=int(mesh_smooth_iterations),
                lambd=float(mesh_smooth_lambda),
            )
    stats["num_vertices"] = float(len(vertices))
    stats["num_faces"] = float(len(faces))

    intermediates = None
    if return_intermediates:
        intermediates = {
            key: value.detach().cpu()
            for key, value in conditioning.items()
            if torch.is_tensor(value)
        }
        intermediates["touch_points"] = touch_points.detach().cpu()
        intermediates["conditioning_touch_points"] = conditioning_touch_points.detach().cpu()
        intermediates["decoder_touch_points"] = decoder_touch_points.detach().cpu()
        intermediates["sdf_volume"] = sdf_values.reshape(resolution, resolution, resolution)
        intermediates["pred_pose_rotation"] = torch.from_numpy(pred_rotation)
        intermediates["used_canonicalization_rotation"] = torch.from_numpy(used_rotation)
        if stage_sdf_chunks is not None:
            for stage_index, chunks in enumerate(stage_sdf_chunks):
                intermediates[f"stage_{stage_index:02d}_sdf_volume"] = torch.cat(chunks).reshape(
                    resolution,
                    resolution,
                    resolution,
                )
    return vertices, faces, stats, intermediates


def object_output_id(path: Path, index: int) -> str:
    digest = hashlib.sha1(path.stem.encode("utf-8")).hexdigest()[:8]
    return f"obj_{index:04d}_{path.stem[:48]}_{digest}"


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = None if args.config is None else load_json(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, resolved_config = load_pose_model(args.checkpoint, config=config, device=device)
    data_dir = Path(args.data_dir).resolve()
    object_files = resolve_split_files(resolved_config, args.split, data_dir)
    if args.max_objects > 0:
        object_files = object_files[: int(args.max_objects)]

    split_cfg = resolved_config.get("dataset", {}).get(args.split, {})
    touch_key = args.touch_key or split_cfg.get("touch_views_key", "touch_points")
    touch_view_indices = parse_index_list(args.touch_view_indices)
    if touch_view_indices is None:
        touch_view_indices = split_cfg.get("touch_view_indices")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(args.checkpoint).resolve().parent / f"pose_head_{args.split}_infer"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "num_objects": len(object_files),
        "merge_touch_views": bool(args.merge_touch_views),
        "oracle_pose": bool(args.oracle_pose),
        "results": [],
        "failures": [],
    }

    for object_index, object_path in enumerate(object_files, start=1):
        print(f"processing {object_path.name}")
        output_id = object_output_id(object_path, object_index)
        object_output_dir = output_dir / output_id
        object_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            if args.merge_touch_views:
                touch_specs = [{"touch_label": "merged", "touch_view_index": None}]
            else:
                touch_specs = list_touch_inputs_from_file(
                    touch_file=object_path,
                    touch_key=touch_key,
                    touch_view_indices=touch_view_indices,
                )
        except Exception as exc:
            summary["failures"].append({"object_file": str(object_path), "stage": "list_touch", "error": str(exc)})
            print(f"  failed to list tactile views: {exc}")
            if args.fail_fast:
                raise
            continue

        for spec in touch_specs:
            label = str(spec.get("touch_label") or "touch")
            if spec.get("touch_view_index") is not None:
                label = f"view{int(spec['touch_view_index']):02d}"
            try:
                if args.merge_touch_views:
                    touch_points, metadata = load_merged_touch_points_from_file(
                        object_path,
                        touch_key=touch_key,
                        touch_view_indices=touch_view_indices,
                    )
                else:
                    touch_points, metadata = load_touch_points_from_file(
                        touch_file=object_path,
                        touch_key=touch_key,
                        touch_view_index=int(spec.get("touch_view_index") or 0),
                        touch_group_key=spec.get("touch_group_key"),
                        touch_group_value=spec.get("touch_group_value"),
                    )
                oracle_rotation = load_oracle_pose_rotation(object_path, device) if args.oracle_pose else None
                vertices, faces, stats, intermediates = infer_pose_mesh(
                    model=model,
                    touch_points=touch_points,
                    resolution=int(args.resolution),
                    chunk_size=int(args.chunk_size),
                    conditioning_touch_point_count=int(args.conditioning_touch_point_count),
                    decoder_touch_point_count=int(args.decoder_touch_point_count),
                    touch_subsample_mode=str(args.touch_subsample_mode),
                    sdf_smooth_sigma=float(args.sdf_smooth_sigma),
                    mesh_decimate_voxel_size=float(args.mesh_decimate_voxel_size),
                    mesh_smooth_method=str(args.mesh_smooth_method),
                    mesh_smooth_iterations=int(args.mesh_smooth_iterations),
                    mesh_smooth_lambda=float(args.mesh_smooth_lambda),
                    mesh_smooth_mu=float(args.mesh_smooth_mu),
                    oracle_rotation=oracle_rotation,
                    return_intermediates=bool(args.save_intermediates),
                )
                mesh_path = object_output_dir / f"{output_id}_{label}.ply"
                preview_path = object_output_dir / f"{output_id}_{label}.png"
                save_mesh_preview(vertices, faces, mesh_path, preview_path, binary_ply=bool(args.binary_ply))
                intermediates_path = None
                if args.save_intermediates and intermediates is not None:
                    intermediates_path = object_output_dir / "intermediates" / f"{output_id}_{label}.npz"
                    save_inference_intermediates(intermediates_path, intermediates, stats, metadata)

                row = {
                    "object_file": str(object_path),
                    "label": label,
                    "mesh_path": str(mesh_path),
                    "preview_path": str(preview_path),
                    "intermediates_path": None if intermediates_path is None else str(intermediates_path),
                    "num_touch_points": int(metadata.get("num_touch_points", 0) or 0),
                    **stats,
                }
                summary["results"].append(row)
                print(
                    f"  {label} | vertices={int(stats['num_vertices'])} faces={int(stats['num_faces'])} "
                    f"sdf=[{stats['sdf_min']:.5f}, {stats['sdf_max']:.5f}] "
                    f"touch_ratio={stats['touch_count_ratio']:.3f}"
                )
            except Exception as exc:
                summary["failures"].append(
                    {"object_file": str(object_path), "view_label": label, "stage": "infer", "error": str(exc)}
                )
                print(f"  {label} failed: {exc}")
                if args.fail_fast:
                    raise

    summary["successful_runs"] = len(summary["results"])
    summary["failed_runs"] = len(summary["failures"])
    summary_path = output_dir / f"{args.split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_csv(output_dir / f"{args.split}_results.csv", summary["results"])
    save_csv(output_dir / f"{args.split}_failures.csv", summary["failures"])
    print(f"saved summary to {summary_path}")


if __name__ == "__main__":
    main()
