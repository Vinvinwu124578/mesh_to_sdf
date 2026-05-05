from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

TACTISTRUCT_ROOT = Path(r"C:\Users\wudaw\Downloads\Tactistruct-main\Tactistruct-main")
sys.path.insert(0, str(TACTISTRUCT_ROOT / "src"))

from tactistruct_progressive_attn_fix.inference_utils import infer_mesh, load_model_from_checkpoint, save_mesh_preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for the progressive_attn_fix optional-normal variant.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--touch-file", type=str, required=True)
    parser.add_argument("--touch-key", type=str, default="touch_points")
    parser.add_argument("--touch-view-index", type=int, default=0)
    parser.add_argument("--touch-group-key", type=str, default=None)
    parser.add_argument("--touch-group-value", type=int, default=None)
    parser.add_argument("--use-touch-point-normals", action="store_true")
    parser.add_argument("--touch-point-normals-key", type=str, default="touch_point_normals")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=32768)
    parser.add_argument("--conditioning-touch-point-count", type=int, default=0)
    parser.add_argument("--decoder-touch-point-count", type=int, default=0)
    parser.add_argument("--touch-subsample-mode", type=str, default="random", choices=["random", "fps"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-mesh", type=str, required=True)
    parser.add_argument("--preview-image", type=str, default=None)
    return parser.parse_args()


def _resolve_touch_array(payload: dict, key: str) -> np.ndarray:
    if key in payload:
        return np.asarray(payload[key], dtype=np.float32)
    raise KeyError(f"Could not find key '{key}' in tactile file. Available keys: {list(payload.keys())}")


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
        touch_points = _resolve_touch_array(payload, touch_key)
        original_ndim = touch_points.ndim
        resolved_group_value = touch_group_value

        normals = None
        if use_touch_point_normals:
            normals = _resolve_touch_array(payload, touch_point_normals_key)
            if normals.shape != touch_points.shape:
                raise ValueError(
                    f"Normal key '{touch_point_normals_key}' is not aligned with '{touch_key}'. "
                    f"Expected shape {touch_points.shape}, got {normals.shape}."
                )

        if touch_points.ndim == 3:
            if touch_view_index < 0 or touch_view_index >= touch_points.shape[0]:
                raise IndexError(
                    f"touch_view_index {touch_view_index} is out of range for key '{touch_key}' "
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
            "touch_key": touch_key,
            "touch_view_index": touch_view_index if original_ndim == 3 else None,
            "touch_group_key": touch_group_key,
            "touch_group_value": resolved_group_value,
            "num_touch_points": int(touch_points.shape[0]),
            "use_touch_point_normals": bool(use_touch_point_normals),
        }
        return torch.from_numpy(touch_points).unsqueeze(0), metadata


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, resolved_config = load_model_from_checkpoint(args.checkpoint, config=None, device=device)
    use_touch_point_normals = bool(args.use_touch_point_normals)
    train_cfg = resolved_config.get("dataset", {}).get("train", {})
    if not use_touch_point_normals:
        use_touch_point_normals = bool(train_cfg.get("use_touch_point_normals", False))
    touch_points, metadata = load_touch_points_from_file_optional_normals(
        touch_file=args.touch_file,
        touch_key=args.touch_key,
        touch_view_index=int(args.touch_view_index),
        touch_group_key=args.touch_group_key,
        touch_group_value=args.touch_group_value,
        use_touch_point_normals=use_touch_point_normals,
        touch_point_normals_key=args.touch_point_normals_key,
    )
    vertices, faces, stats = infer_mesh(
        model=model,
        touch_points=touch_points,
        resolution=int(args.resolution),
        chunk_size=int(args.chunk_size),
        conditioning_touch_point_count=int(args.conditioning_touch_point_count) or None,
        decoder_touch_point_count=int(args.decoder_touch_point_count) or None,
        touch_subsample_mode=str(args.touch_subsample_mode),
        device=device,
    )
    mesh_path = Path(args.output_mesh).resolve()
    preview_path = None if args.preview_image is None else Path(args.preview_image).resolve()
    save_mesh_preview(vertices, faces, mesh_path, preview_path)
    print(
        f"[OK] mesh={mesh_path} verts={int(stats['num_vertices'])} faces={int(stats['num_faces'])} "
        f"touch_points={metadata['num_touch_points']} use_touch_point_normals={metadata['use_touch_point_normals']}"
    )


if __name__ == "__main__":
    main()
