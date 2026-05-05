from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from common import collect_npz_files, load_json, parse_index_list, resolve_split_files, save_json
from dataset import load_adaptive_touch_points_from_file
from tactistruct.utils.checkpoint import load_project_checkpoint
from tactistruct.utils.geometry import create_query_grid, extract_mesh_from_sdf
from tactistruct.utils.ops import farthest_point_sample, gather_points
from tactistruct.inference_utils import save_mesh_preview
from tactistruct_progressive_attn_fix.model import ProgressiveCompletionSystemAttnFix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference for the adaptive-confidence progressive_attn_fix variant without modifying the original package."
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
    parser.add_argument("--touch-subsample-mode", type=str, default="confidence", choices=["random", "fps", "confidence", "confidence_fps"])
    parser.add_argument("--touch-confidence-power", type=float, default=1.5)
    parser.add_argument("--touch-confidence-floor", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    config: dict | None = None,
    device: torch.device | None = None,
) -> tuple[ProgressiveCompletionSystemAttnFix, dict]:
    checkpoint = load_project_checkpoint(checkpoint_path, map_location="cpu")
    resolved_config = checkpoint.get("config") if config is None else config
    if resolved_config is None:
        raise ValueError("Could not resolve config for inference.")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = resolved_config["model"]
    model = ProgressiveCompletionSystemAttnFix(
        encoder_cfg=model_cfg.get("encoder", {}),
        latent_path_cfg=model_cfg.get("latent_path", {}),
        decoder_cfg=model_cfg.get("decoder", {}),
        active_sampling_cfg=model_cfg.get("active_sampling", {}),
        surface_query_samples=int(model_cfg.get("surface_query_samples", 512)),
        use_touch_conditioning=bool(model_cfg.get("use_touch_conditioning", True)),
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
    return resolve_split_files(config["dataset"][args.split])


def list_view_indices(touch_file: Path, touch_view_indices: list[int] | None) -> list[int]:
    with np.load(touch_file, mmap_mode="r") as payload:
        if "patch_points" not in payload:
            raise KeyError(f"Could not find 'patch_points' in {touch_file}.")
        available_count = int(payload["patch_points"].shape[0])
    if touch_view_indices is None:
        return list(range(available_count))
    invalid_indices = [index for index in touch_view_indices if index < 0 or index >= available_count]
    if invalid_indices:
        raise IndexError(f"touch_view_indices contains out-of-range values {invalid_indices}; file only has {available_count} tactile views.")
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


def confidence_weighted_indices(
    touch_points: torch.Tensor,
    max_points: int,
    confidence_power: float,
    confidence_floor: float,
) -> torch.Tensor:
    confidence = touch_points[..., 6].clamp(0.0, 1.0)
    weights = confidence_floor + (1.0 - confidence_floor) * confidence.pow(confidence_power)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return torch.multinomial(weights, num_samples=max_points, replacement=False)


def subsample_touch_points(
    touch_points: torch.Tensor | None,
    max_points: int | None,
    mode: str,
    confidence_power: float,
    confidence_floor: float,
) -> torch.Tensor | None:
    if touch_points is None or max_points is None or max_points <= 0 or touch_points.size(1) <= max_points:
        return touch_points
    resolved_mode = str(mode).lower()
    if resolved_mode == "fps":
        indices = farthest_point_sample(touch_points[..., :3], max_points)
        return gather_points(touch_points, indices)
    if resolved_mode == "random":
        scores = torch.rand(touch_points.size(0), touch_points.size(1), device=touch_points.device, dtype=touch_points.dtype)
        indices = scores.topk(k=max_points, dim=1).indices
        return gather_points(touch_points, indices)
    if resolved_mode == "confidence":
        indices = confidence_weighted_indices(touch_points, max_points, confidence_power, confidence_floor)
        return gather_points(touch_points, indices)
    if resolved_mode == "confidence_fps":
        proposal_count = min(touch_points.size(1), max(max_points * 4, max_points))
        proposal_idx = confidence_weighted_indices(touch_points, proposal_count, confidence_power, confidence_floor)
        proposed = gather_points(touch_points, proposal_idx)
        keep_idx = farthest_point_sample(proposed[..., :3], max_points)
        return gather_points(proposed, keep_idx)
    raise ValueError(f"Unsupported touch subsample mode: {mode}")


def compute_touch_count_ratio(
    conditioning_touch_points: torch.Tensor | None,
    decoder_touch_points: torch.Tensor | None,
    conditioning_touch_point_count: int | None,
    decoder_touch_point_count: int | None,
) -> torch.Tensor | None:
    observed_counts = []
    target_counts = []
    if conditioning_touch_points is not None:
        observed_counts.append(int(conditioning_touch_points.size(1)))
        target_counts.append(int(conditioning_touch_point_count) if conditioning_touch_point_count and conditioning_touch_point_count > 0 else int(conditioning_touch_points.size(1)))
    if decoder_touch_points is not None:
        observed_counts.append(int(decoder_touch_points.size(1)))
        target_counts.append(int(decoder_touch_point_count) if decoder_touch_point_count and decoder_touch_point_count > 0 else int(decoder_touch_points.size(1)))
    if not observed_counts:
        return None
    ratios = [float(obs) / float(max(target, 1)) for obs, target in zip(observed_counts, target_counts)]
    return torch.tensor([[max(0.0, min(1.0, min(ratios)))]], dtype=torch.float32)


def infer_mesh(
    model: ProgressiveCompletionSystemAttnFix,
    touch_points: torch.Tensor | None,
    resolution: int,
    chunk_size: int,
    conditioning_touch_point_count: int | None,
    decoder_touch_point_count: int | None,
    touch_subsample_mode: str,
    confidence_power: float,
    confidence_floor: float,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if device is None:
        device = next(model.parameters()).device
    batch_size = 1
    if touch_points is not None:
        touch_points = touch_points.to(device)
        batch_size = int(touch_points.size(0))
    if batch_size != 1:
        raise ValueError("infer_mesh currently expects exactly one tactile sample at a time.")

    conditioning_touch_points = subsample_touch_points(touch_points, conditioning_touch_point_count, touch_subsample_mode, confidence_power, confidence_floor)
    decoder_touch_points = subsample_touch_points(touch_points, decoder_touch_point_count, touch_subsample_mode, confidence_power, confidence_floor)
    decoder_touch_xyz = None if decoder_touch_points is None else decoder_touch_points[..., :3]
    touch_count_ratio = compute_touch_count_ratio(
        conditioning_touch_points=conditioning_touch_points,
        decoder_touch_points=decoder_touch_points,
        conditioning_touch_point_count=conditioning_touch_point_count,
        decoder_touch_point_count=decoder_touch_point_count,
    )

    grid = create_query_grid(resolution).to(device)
    final_sdf_chunks = []
    with torch.no_grad():
        conditioning_outputs = model.encode_touch(
            batch_size=batch_size,
            device=device,
            touch_points=conditioning_touch_points,
            apply_augmentation=False,
            stochastic=False,
        )
        for start in range(0, grid.size(0), chunk_size):
            chunk = grid[start : start + chunk_size].unsqueeze(0).expand(batch_size, -1, -1)
            stage_sdf, _ = model.decode_points(
                chunk,
                conditioning_outputs["patch_tokens"],
                conditioning_outputs["latent_path"],
                touch_points=decoder_touch_xyz,
                touch_count_ratio=touch_count_ratio,
            )
            final_sdf_chunks.append(stage_sdf[:, -1].squeeze(0).cpu())

    sdf_values = torch.cat(final_sdf_chunks, dim=0)
    stats = {
        "sdf_min": float(sdf_values.min().item()),
        "sdf_max": float(sdf_values.max().item()),
        "sdf_mean": float(sdf_values.mean().item()),
        "touch_count_ratio": 0.0 if touch_count_ratio is None else float(touch_count_ratio.item()),
        "mean_touch_confidence": 0.0 if touch_points is None else float(touch_points[..., 6].mean().item()),
    }
    if not (stats["sdf_min"] <= 0.0 <= stats["sdf_max"]):
        raise RuntimeError(
            "Predicted SDF does not cross zero on the query grid, so marching cubes cannot recover a valid surface. "
            f"Observed range: [{stats['sdf_min']:.6f}, {stats['sdf_max']:.6f}]"
        )
    vertices, faces = extract_mesh_from_sdf(sdf_values, resolution)
    stats["num_vertices"] = float(len(vertices))
    stats["num_faces"] = float(len(faces))
    return vertices, faces, stats


def main() -> None:
    args = parse_args()
    config = None if args.config is None else load_json(args.config)
    device = torch.device(args.device)
    model, resolved_config = load_model_from_checkpoint(args.checkpoint, config=config, device=device)

    files = resolve_input_files(args, resolved_config)
    if args.max_objects > 0:
        files = files[: int(args.max_objects)]

    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else Path(args.checkpoint).resolve().parent / "inference_adaptive_confidence"
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
                mesh_name = str(payload["mesh_name"].item()) if "mesh_name" in payload and np.asarray(payload["mesh_name"]).shape == () else object_path.stem
                coverage = float(payload["planning_surface_coverage_ratio"]) if "planning_surface_coverage_ratio" in payload else None

            resolved_views = list_view_indices(object_path, touch_view_indices)
            touch_specs = [("merged", resolved_views)] if args.merge_touch_views else [(f"view{int(view_index):02d}", [int(view_index)]) for view_index in resolved_views]

            for label, selected_views in touch_specs:
                touch_points, touch_band_width, metadata = load_adaptive_touch_points_from_file(object_path, touch_view_indices=selected_views)
                vertices, faces, stats = infer_mesh(
                    model=model,
                    touch_points=touch_points,
                    resolution=int(args.resolution),
                    chunk_size=int(args.chunk_size),
                    conditioning_touch_point_count=conditioning_touch_point_count,
                    decoder_touch_point_count=decoder_touch_point_count,
                    touch_subsample_mode=str(args.touch_subsample_mode),
                    confidence_power=float(args.touch_confidence_power),
                    confidence_floor=float(args.touch_confidence_floor),
                    device=device,
                )
                mesh_path = object_output_dir / f"{label}.ply"
                preview_path = object_output_dir / f"{label}.png" if bool(args.save_preview) else None
                save_mesh_preview(vertices, faces, mesh_path, preview_path)
                rows.append(
                    {
                        "object_path": str(object_path),
                        "mesh_name": mesh_name,
                        "touch_label": label,
                        "touch_view_indices": selected_views,
                        "mesh_path": str(mesh_path),
                        "planning_surface_coverage_ratio": coverage,
                        "num_touch_points": int(metadata["num_touch_points"]),
                        "mean_touch_band_width": float(touch_band_width.mean().item()),
                        **stats,
                    }
                )
                print(f"[OK] {object_path.name} touch={label} verts={int(stats['num_vertices'])} faces={int(stats['num_faces'])}")
        except Exception as exc:
            failures.append({"object_path": str(object_path), "error": str(exc)})
            print(f"[FAILED] {object_path}")
            print(exc)
            if bool(args.fail_fast):
                raise

    summary = {"checkpoint": str(Path(args.checkpoint).resolve()), "num_objects": len(files), "num_meshes": len(rows), "failures": failures, "rows": rows}
    summary_path = save_json(output_dir / "inference_summary.json", summary)
    print(f"[SUMMARY] saved to {summary_path}")


if __name__ == "__main__":
    main()
