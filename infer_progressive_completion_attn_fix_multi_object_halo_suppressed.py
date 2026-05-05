from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import torch

TACTISTRUCT_ROOT = Path(r"C:\Users\wudaw\Downloads\Tactistruct-main\Tactistruct-main")
sys.path.insert(0, str(TACTISTRUCT_ROOT / "src"))

from tactistruct_progressive_attn_fix.inference_utils import (
    infer_mesh,
    list_touch_inputs_from_file,
    load_model_from_checkpoint,
    load_touch_points_from_file,
    save_inference_intermediates,
    save_mesh_preview,
)
from tactistruct.utils.metrics import compute_reconstruction_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch inference for the halo-suppressed progressive tactile completion project."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--touch-key", type=str, default=None)
    parser.add_argument("--touch-group-key", type=str, default=None)
    parser.add_argument("--touch-view-indices", type=str, default=None)
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=32768)
    parser.add_argument("--conditioning-touch-point-count", type=int, default=0)
    parser.add_argument("--decoder-touch-point-count", type=int, default=0)
    parser.add_argument("--touch-subsample-mode", type=str, default="random", choices=["random", "fps"])
    parser.add_argument("--merge-touch-views", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--component-filter-mode",
        type=str,
        default="none",
        choices=["none", "largest", "closest_touch", "hybrid"],
        help="Connected-component post-filter applied after marching cubes. Default is disabled.",
    )
    parser.add_argument(
        "--boundary-tol-ratio",
        type=float,
        default=2.0,
        help="How close a component can get to the query-grid boundary before being treated as a boundary artifact.",
    )
    parser.add_argument(
        "--component-distance-sample-vertices",
        type=int,
        default=2048,
        help="How many vertices per component to sample when scoring distance to touch points.",
    )
    parser.add_argument("--compute-cd", action="store_true")
    parser.add_argument("--compute-emd", action="store_true")
    parser.add_argument("--compute-fscore", action="store_true")
    parser.add_argument("--compute-silhouette-iou", action="store_true")
    parser.add_argument("--fscore-threshold", type=float, default=0.01)
    parser.add_argument("--metric-samples", type=int, default=512)
    parser.add_argument("--metric-seed", type=int, default=42)
    parser.add_argument("--emd-sinkhorn-reg", type=float, default=0.1)
    parser.add_argument("--emd-sinkhorn-iters", type=int, default=100)
    parser.add_argument("--silhouette-image-size", type=int, default=128)
    parser.add_argument("--save-intermediates", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def component_touches_boundary(vertices: np.ndarray, bound: float, tolerance: float) -> bool:
    if vertices.size == 0:
        return False
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    return bool(np.any(mins <= (-bound + tolerance)) or np.any(maxs >= (bound - tolerance)))


def compute_component_touch_distance(
    vertices: np.ndarray,
    touch_points: torch.Tensor | None,
    sample_vertices: int,
) -> float:
    if touch_points is None or vertices.size == 0:
        return float("inf")
    touch_xyz = touch_points[0, :, :3].detach().cpu().float()
    if touch_xyz.numel() == 0:
        return float("inf")

    if sample_vertices > 0 and len(vertices) > sample_vertices:
        indices = np.linspace(0, len(vertices) - 1, num=sample_vertices, dtype=np.int64)
        sampled_vertices = vertices[indices]
    else:
        sampled_vertices = vertices
    vertex_tensor = torch.from_numpy(sampled_vertices.astype(np.float32))
    distances = torch.cdist(touch_xyz, vertex_tensor).min(dim=1).values
    return float(distances.mean().item())


def filter_mesh_components(
    vertices: np.ndarray,
    faces: np.ndarray,
    touch_points: torch.Tensor | None,
    resolution: int,
    mode: str,
    boundary_tol_ratio: float,
    distance_sample_vertices: int,
    bound: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    info = {
        "component_count": 1.0,
        "boundary_filtered_count": 0.0,
        "selected_component_faces": float(len(faces)),
        "selected_component_touch_distance": 0.0,
    }
    if str(mode).lower() == "none" or len(faces) == 0:
        return vertices, faces, info

    try:
        import trimesh
    except ImportError:
        warnings.warn(
            "trimesh is not available, so component filtering was skipped.",
            RuntimeWarning,
            stacklevel=2,
        )
        return vertices, faces, info

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    components = list(mesh.split(only_watertight=False))
    if len(components) <= 1:
        return vertices, faces, info

    step = (2.0 * bound) / max(int(resolution) - 1, 1)
    boundary_tol = float(boundary_tol_ratio) * step
    component_infos = []
    for comp_index, component in enumerate(components):
        comp_vertices = np.asarray(component.vertices, dtype=np.float32)
        comp_faces = np.asarray(component.faces, dtype=np.int32)
        touches_boundary = component_touches_boundary(comp_vertices, bound=bound, tolerance=boundary_tol)
        touch_distance = compute_component_touch_distance(
            comp_vertices,
            touch_points=touch_points,
            sample_vertices=int(distance_sample_vertices),
        )
        component_infos.append(
            {
                "index": comp_index,
                "vertices": comp_vertices,
                "faces": comp_faces,
                "touches_boundary": touches_boundary,
                "touch_distance": touch_distance,
                "num_faces": int(len(comp_faces)),
                "num_vertices": int(len(comp_vertices)),
            }
        )

    info["component_count"] = float(len(component_infos))
    non_boundary = [item for item in component_infos if not item["touches_boundary"]]
    info["boundary_filtered_count"] = float(len(component_infos) - len(non_boundary))
    candidate_components = non_boundary if non_boundary else component_infos

    resolved_mode = str(mode).lower()
    if resolved_mode == "largest":
        best_component = max(candidate_components, key=lambda item: (item["num_faces"], item["num_vertices"]))
    elif resolved_mode == "closest_touch":
        best_component = min(candidate_components, key=lambda item: (item["touch_distance"], -item["num_faces"]))
    else:
        if all(not np.isfinite(item["touch_distance"]) for item in candidate_components):
            best_component = max(candidate_components, key=lambda item: (item["num_faces"], item["num_vertices"]))
        else:
            best_distance = min(item["touch_distance"] for item in candidate_components if np.isfinite(item["touch_distance"]))
            distance_slack = max(step * 4.0, 0.15 * max(best_distance, step))
            close_components = [
                item for item in candidate_components if item["touch_distance"] <= best_distance + distance_slack
            ]
            best_component = max(close_components, key=lambda item: (item["num_faces"], item["num_vertices"]))

    info["selected_component_faces"] = float(best_component["num_faces"])
    info["selected_component_touch_distance"] = float(best_component["touch_distance"])
    return best_component["vertices"], best_component["faces"], info


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


def build_view_label(metadata: dict[str, int | str | None]) -> str:
    if metadata.get("touch_label") is not None:
        return str(metadata["touch_label"])
    if metadata.get("touch_view_index") is not None:
        return f"view{int(metadata['touch_view_index']):02d}"
    if metadata.get("touch_group_value") is not None:
        return f"group{int(metadata['touch_group_value'])}"
    return "touch"


def build_object_output_id(object_path: Path, object_index: int) -> str:
    stem = object_path.stem
    category = stem.split("__")[0] if "__" in stem else stem.split("_")[0]
    category = "".join(ch for ch in category if ch.isalnum()).lower() or "object"
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:10]
    return f"obj_{object_index:04d}_{category}_{digest}"


def resolve_split_files(split_cfg: dict) -> list[Path]:
    if split_cfg.get("name", "").lower() != "npz":
        raise ValueError("Attention batch inference currently expects an npz split config.")

    data_dir = (Path(split_cfg["root"]) / split_cfg.get("split", ".")).resolve()
    recursive = bool(split_cfg.get("recursive", False))
    all_files = sorted(data_dir.rglob("*.npz") if recursive else data_dir.glob("*.npz"))
    if not all_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    if split_cfg.get("object_filenames"):
        relative_map = {path.relative_to(data_dir).as_posix(): path for path in all_files}
        basename_map: dict[str, list[Path]] = {}
        for path in all_files:
            basename_map.setdefault(path.name, []).append(path)

        selected_files = []
        missing: list[str] = []
        ambiguous: dict[str, list[str]] = {}
        for name in split_cfg["object_filenames"]:
            normalized = Path(name).as_posix().lstrip("./")
            if normalized in relative_map:
                selected_files.append(relative_map[normalized])
                continue

            basename = Path(normalized).name
            matches = basename_map.get(basename, [])
            if not matches:
                missing.append(name)
                continue
            if len(matches) > 1:
                ambiguous[name] = [match.relative_to(data_dir).as_posix() for match in matches]
                continue
            selected_files.append(matches[0])

        if ambiguous:
            raise FileNotFoundError(
                "Split config referenced ambiguous object files; use relative paths from the dataset root. "
                f"Ambiguous entries: {ambiguous}"
            )
        if missing:
            if selected_files:
                warnings.warn(
                    "Split config referenced some object files that are not present in the current data directory. "
                    f"Proceeding with the {len(selected_files)} files that were found and skipping {len(missing)} "
                    f"missing entries.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return selected_files
            warnings.warn(
                "Split config referenced only missing object files for the current data directory. "
                f"Falling back to all {len(all_files)} available .npz files under {data_dir}.",
                RuntimeWarning,
                stacklevel=2,
            )
            return all_files
        return selected_files

    if split_cfg.get("object_indices"):
        return [all_files[int(index)] for index in split_cfg["object_indices"]]

    return all_files


def save_summary(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _collect_csv_fieldnames(rows: list[dict]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def save_csv_table(path: Path, rows: list[dict]) -> Path | None:
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _collect_csv_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def load_reference_surface_points(object_path: Path) -> np.ndarray:
    with np.load(object_path) as payload:
        if "surface_points" not in payload:
            raise KeyError(f"Could not find 'surface_points' in {object_path}.")
        return payload["surface_points"].astype(np.float32)


def load_reference_occupancy_points(object_path: Path) -> np.ndarray:
    with np.load(object_path) as payload:
        if "query_points" not in payload or "query_sdf" not in payload:
            raise KeyError(f"Could not find 'query_points' and 'query_sdf' in {object_path}.")

        query_points = payload["query_points"].astype(np.float32)
        query_sdf = payload["query_sdf"].astype(np.float32)
        if query_points.ndim != 2 or query_points.shape[1] != 3:
            raise ValueError(f"'query_points' must have shape [N, 3] in {object_path}, got {query_points.shape}.")
        if query_sdf.ndim != 1 or query_sdf.shape[0] != query_points.shape[0]:
            raise ValueError(
                f"'query_sdf' must have shape [N] aligned with query_points in {object_path}, "
                f"got {query_sdf.shape} vs {query_points.shape}."
            )

        inside_mask = query_sdf <= 0.0
        if not np.any(inside_mask):
            raise ValueError(f"No inside points with query_sdf <= 0 were found in {object_path}.")
        return query_points[inside_mask]


def load_merged_touch_points_from_file(
    touch_file: str | Path,
    touch_key: str = "touch_points",
    touch_view_indices: list[int] | None = None,
) -> tuple[torch.Tensor, dict[str, int | str | None]]:
    with np.load(touch_file) as payload:
        resolved_touch_key = touch_key
        if resolved_touch_key not in payload:
            if "touch_point_sets" in payload:
                resolved_touch_key = "touch_point_sets"
            else:
                raise KeyError(
                    f"Could not find tactile key '{touch_key}' in {touch_file}. "
                    f"Available keys: {list(payload.keys())}"
                )

        touch_points = payload[resolved_touch_key].astype(np.float32)
        if touch_points.ndim != 3:
            raise ValueError(
                f"Merged tactile inference expects a [V, N, C] tensor in key '{resolved_touch_key}', "
                f"but received shape {touch_points.shape}."
            )

        available_count = int(touch_points.shape[0])
        selected_indices = list(range(available_count)) if touch_view_indices is None else [int(index) for index in touch_view_indices]
        invalid_indices = [index for index in selected_indices if index < 0 or index >= available_count]
        if invalid_indices:
            raise IndexError(
                f"touch_view_indices contains out-of-range values {invalid_indices} "
                f"for {touch_file}, which has {available_count} tactile views."
            )

        merged_touch_points = touch_points[selected_indices].reshape(-1, touch_points.shape[-1])
        metadata = {
            "touch_key": resolved_touch_key,
            "touch_view_index": None,
            "touch_group_key": None,
            "touch_group_value": None,
            "touch_label": "merged",
            "num_touch_points": int(merged_touch_points.shape[0]),
        }
        return torch.from_numpy(merged_touch_points).unsqueeze(0), metadata


def summarise_metric_results(results: list[dict]) -> dict[str, float]:
    metric_keys = ("cd_l1", "cd_l2", "emd", "fscore", "fscore_precision", "fscore_recall", "silhouette_iou")
    summary: dict[str, float] = {}
    for key in metric_keys:
        values = [float(item[key]) for item in results if key in item]
        if values:
            summary[f"mean_{key}"] = float(sum(values) / len(values))
    return summary


def _resolve_touch_point_count(requested_count: int, split_cfg: dict) -> int | None:
    if requested_count > 0:
        return int(requested_count)
    default_count = int(split_cfg.get("num_touch_points", 512))
    return default_count if default_count > 0 else None


def main() -> None:
    args = parse_args()
    config = None if args.config is None else load_json(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, resolved_config = load_model_from_checkpoint(args.checkpoint, config=config, device=device)

    split_cfg = resolved_config.get("dataset", {}).get(args.split)
    if split_cfg is None:
        raise KeyError(f"Could not find dataset split '{args.split}' in the resolved config.")

    object_files = resolve_split_files(split_cfg)
    if args.max_objects > 0:
        object_files = object_files[: args.max_objects]

    default_output_dir = Path(args.checkpoint).resolve().parent / f"progressive_attn_fix_halo_suppressed_{args.split}_infer"
    output_dir = Path(args.output_dir) if args.output_dir is not None else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    touch_key = args.touch_key or split_cfg.get("touch_views_key", "touch_points")
    touch_group_key = args.touch_group_key if args.touch_group_key is not None else split_cfg.get("touch_group_key")
    touch_view_indices = parse_index_list(args.touch_view_indices)
    if touch_view_indices is None:
        touch_view_indices = split_cfg.get("touch_view_indices")

    conditioning_touch_point_count = _resolve_touch_point_count(args.conditioning_touch_point_count, split_cfg)
    if args.decoder_touch_point_count > 0:
        decoder_touch_point_count = int(args.decoder_touch_point_count)
    else:
        decoder_touch_point_count = conditioning_touch_point_count

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "num_objects": len(object_files),
        "merge_touch_views": bool(args.merge_touch_views),
        "touch_subsample_mode": str(args.touch_subsample_mode),
        "component_filter_mode": str(args.component_filter_mode),
        "boundary_tol_ratio": float(args.boundary_tol_ratio),
        "conditioning_touch_point_count": conditioning_touch_point_count,
        "decoder_touch_point_count": decoder_touch_point_count,
        "metrics": {
            "compute_cd": bool(args.compute_cd or args.compute_emd or args.compute_fscore),
            "compute_emd": bool(args.compute_emd),
            "compute_fscore": bool(args.compute_fscore),
            "compute_silhouette_iou": bool(args.compute_silhouette_iou),
            "fscore_threshold": float(args.fscore_threshold),
            "metric_samples": int(args.metric_samples),
            "emd_sinkhorn_reg": float(args.emd_sinkhorn_reg),
            "emd_sinkhorn_iters": int(args.emd_sinkhorn_iters),
            "silhouette_image_size": int(args.silhouette_image_size),
        },
        "results": [],
        "failures": [],
    }

    for object_index, object_path in enumerate(object_files, start=1):
        print(f"processing object: {object_path.name}")
        object_id = build_object_output_id(object_path, object_index)
        object_output_dir = output_dir / object_id
        object_output_dir.mkdir(parents=True, exist_ok=True)
        reference_surface_points = None
        reference_occupancy_points = None
        if args.compute_cd or args.compute_emd or args.compute_fscore:
            try:
                reference_surface_points = load_reference_surface_points(object_path)
            except Exception as exc:
                summary["failures"].append(
                    {"object_file": str(object_path), "stage": "load_reference_surface", "error": str(exc)}
                )
                print(f"  failed to load reference surface points: {exc}")
                if args.fail_fast:
                    raise
                continue
        if args.compute_silhouette_iou:
            try:
                reference_occupancy_points = load_reference_occupancy_points(object_path)
            except Exception as exc:
                summary["failures"].append(
                    {"object_file": str(object_path), "stage": "load_reference_occupancy", "error": str(exc)}
                )
                print(f"  failed to load reference occupancy points: {exc}")
                if args.fail_fast:
                    raise
                continue

        if args.merge_touch_views:
            touch_specs = [
                {
                    "touch_key": touch_key,
                    "touch_view_index": None,
                    "touch_group_key": None,
                    "touch_group_value": None,
                    "touch_label": "merged",
                }
            ]
        else:
            try:
                touch_specs = list_touch_inputs_from_file(
                    touch_file=object_path,
                    touch_key=touch_key,
                    touch_group_key=touch_group_key,
                    touch_view_indices=touch_view_indices,
                )
            except Exception as exc:
                summary["failures"].append(
                    {"object_file": str(object_path), "stage": "list_touch_inputs", "error": str(exc)}
                )
                print(f"  failed to enumerate tactile inputs: {exc}")
                if args.fail_fast:
                    raise
                continue

        for spec in touch_specs:
            label = "unknown"
            try:
                if args.merge_touch_views:
                    touch_points, metadata = load_merged_touch_points_from_file(
                        touch_file=object_path,
                        touch_key=str(spec["touch_key"]),
                        touch_view_indices=touch_view_indices,
                    )
                else:
                    touch_points, metadata = load_touch_points_from_file(
                        touch_file=object_path,
                        touch_key=str(spec["touch_key"]),
                        touch_view_index=int(spec["touch_view_index"] or 0),
                        touch_group_key=spec["touch_group_key"],
                        touch_group_value=spec["touch_group_value"],
                    )
                label = build_view_label(metadata)
                mesh_path = object_output_dir / f"{object_id}_{label}.ply"
                preview_path = object_output_dir / f"{object_id}_{label}.png"
                intermediates_path = None
                if args.save_intermediates:
                    intermediates_path = object_output_dir / "intermediates" / f"{object_id}_{label}_intermediates.npz"

                result = infer_mesh(
                    model=model,
                    touch_points=touch_points,
                    resolution=args.resolution,
                    chunk_size=args.chunk_size,
                    conditioning_touch_point_count=conditioning_touch_point_count,
                    decoder_touch_point_count=decoder_touch_point_count,
                    touch_subsample_mode=str(args.touch_subsample_mode),
                    device=device,
                    return_intermediates=bool(intermediates_path),
                )
                if intermediates_path is None:
                    vertices, faces, stats = result
                    intermediates = None
                else:
                    vertices, faces, stats, intermediates = result

                filtered_vertices, filtered_faces, filter_info = filter_mesh_components(
                    vertices=vertices,
                    faces=faces,
                    touch_points=touch_points,
                    resolution=int(args.resolution),
                    mode=str(args.component_filter_mode),
                    boundary_tol_ratio=float(args.boundary_tol_ratio),
                    distance_sample_vertices=int(args.component_distance_sample_vertices),
                )
                vertices, faces = filtered_vertices, filtered_faces
                stats["num_vertices"] = float(len(vertices))
                stats["num_faces"] = float(len(faces))
                stats.update(filter_info)

                saved_preview = save_mesh_preview(
                    vertices=vertices,
                    faces=faces,
                    output_path=mesh_path,
                    preview_image_path=preview_path,
                )
                saved_intermediates = None
                if intermediates_path is not None and intermediates is not None:
                    saved_intermediates = save_inference_intermediates(
                        path=intermediates_path,
                        intermediates=intermediates,
                        stats=stats,
                        metadata=metadata,
                    )

                metric_payload: dict[str, float] = {}
                if reference_surface_points is not None or reference_occupancy_points is not None:
                    metric_seed_material = f"{object_path.as_posix()}::{label}::{args.metric_seed}"
                    metric_seed = int(hashlib.sha1(metric_seed_material.encode("utf-8")).hexdigest()[:8], 16)
                    metric_rng = np.random.default_rng(metric_seed)
                    metric_payload = compute_reconstruction_metrics(
                        vertices=vertices,
                        faces=faces,
                        reference_surface_points=reference_surface_points,
                        sample_count=int(args.metric_samples),
                        rng=metric_rng,
                        compute_cd_flag=bool(args.compute_cd or args.compute_emd or args.compute_fscore),
                        compute_emd=bool(args.compute_emd),
                        compute_fscore_flag=bool(args.compute_fscore),
                        reference_occupancy_points=reference_occupancy_points,
                        compute_silhouette_iou_flag=bool(args.compute_silhouette_iou),
                        silhouette_image_size=int(args.silhouette_image_size),
                        fscore_threshold=float(args.fscore_threshold),
                        emd_reg=float(args.emd_sinkhorn_reg),
                        emd_iters=int(args.emd_sinkhorn_iters),
                    )

                summary["results"].append(
                    {
                        "object_file": str(object_path),
                        "object_output_id": object_id,
                        "view_label": label,
                        "touch_view_index": metadata.get("touch_view_index"),
                        "touch_group_value": metadata.get("touch_group_value"),
                        "num_touch_points": int(metadata.get("num_touch_points", 0) or 0),
                        "conditioning_touch_point_count": 0
                        if conditioning_touch_point_count is None
                        else int(conditioning_touch_point_count),
                        "decoder_touch_point_count": 0
                        if decoder_touch_point_count is None
                        else int(decoder_touch_point_count),
                        "touch_count_ratio": float(stats.get("touch_count_ratio", 0.0)),
                        "sdf_min": float(stats["sdf_min"]),
                        "sdf_max": float(stats["sdf_max"]),
                        "sdf_mean": float(stats["sdf_mean"]),
                        "num_vertices": int(stats["num_vertices"]),
                        "num_faces": int(stats["num_faces"]),
                        "component_count": int(stats.get("component_count", 1.0)),
                        "boundary_filtered_count": int(stats.get("boundary_filtered_count", 0.0)),
                        "selected_component_faces": int(stats.get("selected_component_faces", stats["num_faces"])),
                        "selected_component_touch_distance": float(stats.get("selected_component_touch_distance", 0.0)),
                        "mesh_path": str(mesh_path),
                        "preview_path": str(saved_preview) if saved_preview is not None else None,
                        "intermediates_path": str(saved_intermediates) if saved_intermediates is not None else None,
                        **metric_payload,
                    }
                )
                metric_text = ""
                if "cd_l1" in metric_payload:
                    metric_text += f" | cd_l1={metric_payload['cd_l1']:.6f}"
                if "fscore" in metric_payload:
                    metric_text += f" | fscore={metric_payload['fscore']:.6f}"
                if "emd" in metric_payload:
                    metric_text += f" | emd={metric_payload['emd']:.6f}"
                if "silhouette_iou" in metric_payload:
                    metric_text += f" | sil_iou={metric_payload['silhouette_iou']:.6f}"
                filter_text = ""
                if str(args.component_filter_mode).lower() != "none":
                    filter_text = (
                        f" | comps={int(stats.get('component_count', 1.0))} "
                        f"| sel_touch_dist={float(stats.get('selected_component_touch_distance', 0.0)):.4f}"
                    )
                print(
                    f"  {label} | vertices={int(stats['num_vertices'])} faces={int(stats['num_faces'])} "
                    f"sdf=[{stats['sdf_min']:.5f}, {stats['sdf_max']:.5f}] "
                    f"| touch_ratio={float(stats.get('touch_count_ratio', 0.0)):.3f} "
                    f"{filter_text}{metric_text}"
                )
            except Exception as exc:
                summary["failures"].append(
                    {
                        "object_file": str(object_path),
                        "view_label": label,
                        "stage": "infer",
                        "error": str(exc),
                    }
                )
                print(f"  {label} failed: {exc}")
                if args.fail_fast:
                    raise

    summary["successful_runs"] = len(summary["results"])
    summary["failed_runs"] = len(summary["failures"])
    summary["metric_summary"] = summarise_metric_results(summary["results"])
    summary_path = save_summary(output_dir / f"{args.split}_summary.json", summary)
    metrics_csv_path = save_csv_table(output_dir / f"{args.split}_metrics.csv", summary["results"])
    metric_summary_rows = [{"metric": key, "value": value} for key, value in summary["metric_summary"].items()]
    metric_summary_csv_path = save_csv_table(output_dir / f"{args.split}_metric_summary.csv", metric_summary_rows)
    failures_csv_path = save_csv_table(output_dir / f"{args.split}_failures.csv", summary["failures"])
    print(f"saved summary to {summary_path}")
    if metrics_csv_path is not None:
        print(f"saved metrics csv to {metrics_csv_path}")
    if metric_summary_csv_path is not None:
        print(f"saved metric summary csv to {metric_summary_csv_path}")
    if failures_csv_path is not None:
        print(f"saved failures csv to {failures_csv_path}")


if __name__ == "__main__":
    main()
