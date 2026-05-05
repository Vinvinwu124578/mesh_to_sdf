from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import traceback
from pathlib import Path

import numpy as np

from common import DEFAULT_SHAPENET_ROOT, save_json


DEFAULT_OUTPUT_FOLDER = "tactistruct_structured_mujoco_coverage_no_pose_onefolder"
DEFAULT_ASSET_FOLDER = "tactistruct_structured_mujoco_coverage_no_pose_assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run coverage-aware MuJoCo tactile preprocessing in parallel and save "
            "structured oracle-style patch tensors without any pose fields."
        )
    )
    parser.add_argument("--root-dir", type=str, default=str(DEFAULT_SHAPENET_ROOT))
    parser.add_argument("--category-names", type=str, default=None)
    parser.add_argument("--max-objects-per-category", type=int, default=275)
    parser.add_argument("--output-folder-name", type=str, default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--asset-folder-name", type=str, default=DEFAULT_ASSET_FOLDER)
    parser.add_argument("--max-workers", type=int, default=max(1, min(4, (os.cpu_count() or 1))))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--num-tactile-samples", type=int, default=10)
    parser.add_argument("--tactile-num-fingers", type=int, default=10)
    parser.add_argument("--tactile-points-per-finger", type=int, default=3000)
    parser.add_argument("--structured-points-per-patch", type=int, default=128)
    parser.add_argument("--dense-surface-sample-n", type=int, default=120000)
    parser.add_argument("--candidate-touch-samples", type=int, default=6000)
    parser.add_argument("--tactile-patch-radius-ratio", type=float, default=0.10)
    parser.add_argument("--tactile-min-touch-separation-ratio", type=float, default=0.055)
    parser.add_argument("--normalization-bound", type=float, default=0.95)
    parser.add_argument("--num-surface-points", type=int, default=235000)
    parser.add_argument("--num-query-points", type=int, default=250000)
    parser.add_argument("--base-seed", type=int, default=42)
    return parser.parse_args()


def parse_category_names(value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [item.strip() for item in value.split(",")]
    names = [item for item in names if item]
    return names or None


def object_seed(base_seed: int, obj_path: str) -> int:
    digest = hashlib.sha1(obj_path.encode("utf-8")).hexdigest()[:8]
    return int(base_seed) + int(digest, 16)


def build_flat_output_path(obj_path: str, root_dir: str, output_folder_name: str) -> str:
    rel_path = os.path.relpath(obj_path, root_dir)
    rel_without_ext = os.path.splitext(rel_path)[0]
    safe_name = rel_without_ext.replace("\\", "__").replace("/", "__").replace(":", "_")
    out_dir = os.path.join(root_dir, output_folder_name)
    return os.path.join(out_dir, safe_name + ".npz")


def build_asset_export_path(obj_path: str, root_dir: str, asset_folder_name: str) -> str:
    rel_path = os.path.relpath(obj_path, root_dir)
    rel_without_ext = os.path.splitext(rel_path)[0]
    safe_name = rel_without_ext.replace("\\", "__").replace("/", "__").replace(":", "_")
    asset_dir = os.path.join(root_dir, asset_folder_name)
    return os.path.join(asset_dir, safe_name + "__normalized.stl")


def subsample_or_repeat(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0:
        raise ValueError("count must be positive.")
    if len(points) == 0:
        return np.zeros((count, 3), dtype=np.float32)
    if len(points) >= count:
        indices = rng.choice(len(points), size=count, replace=False)
        return points[indices].astype(np.float32)
    extra = rng.choice(len(points), size=count - len(points), replace=True)
    merged = np.concatenate([points, points[extra]], axis=0)
    return merged.astype(np.float32)


def build_complete_centers(
    touch_points: np.ndarray,
    finger_ids: np.ndarray,
    touch_centers: np.ndarray,
) -> np.ndarray:
    inferred_fingers = int(finger_ids.max()) + 1 if finger_ids.size > 0 else 0
    num_fingers = max(int(touch_centers.shape[0]), inferred_fingers)
    centers = np.zeros((num_fingers, 3), dtype=np.float32)
    if touch_centers.shape[0] > 0:
        centers[: touch_centers.shape[0]] = touch_centers.astype(np.float32)
    for finger_index in range(num_fingers):
        if finger_index < touch_centers.shape[0]:
            continue
        finger_points = touch_points[finger_ids == finger_index]
        if len(finger_points) > 0:
            centers[finger_index] = finger_points.mean(axis=0).astype(np.float32)
    return centers


def fill_missing_radii(patch_radii: np.ndarray, patch_mask: np.ndarray) -> np.ndarray:
    filled = patch_radii.astype(np.float32).copy()
    valid_values = filled[patch_mask]
    global_default = float(np.median(valid_values)) if valid_values.size > 0 else 1.0
    global_default = max(global_default, 1e-3)
    for finger_index in range(filled.shape[0]):
        finger_valid = patch_mask[finger_index]
        finger_values = filled[finger_index][finger_valid]
        finger_default = float(np.median(finger_values)) if finger_values.size > 0 else global_default
        finger_default = max(finger_default, 1e-3)
        for round_index in range(filled.shape[1]):
            if not patch_mask[finger_index, round_index] or filled[finger_index, round_index] <= 0.0:
                filled[finger_index, round_index] = finger_default
    return np.clip(filled, 1e-3, None).astype(np.float32)


def structure_single_view(
    touch_points: np.ndarray,
    round_ids: np.ndarray,
    finger_ids: np.ndarray,
    touch_centers: np.ndarray,
    points_per_patch: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    touch_points = np.asarray(touch_points, dtype=np.float32)
    round_ids = np.asarray(round_ids, dtype=np.int32).reshape(-1)
    finger_ids = np.asarray(finger_ids, dtype=np.int32).reshape(-1)
    touch_centers = np.asarray(touch_centers, dtype=np.float32)

    centers = build_complete_centers(touch_points, finger_ids, touch_centers)
    num_fingers = int(centers.shape[0])
    num_rounds = int(round_ids.max()) + 1 if round_ids.size > 0 else 0
    if num_fingers <= 0 or num_rounds <= 0:
        raise ValueError("Could not infer valid finger/round structure.")

    patch_points = np.zeros((num_fingers, num_rounds, points_per_patch, 3), dtype=np.float32)
    patch_mask = np.zeros((num_fingers, num_rounds), dtype=bool)
    patch_counts = np.zeros((num_fingers, num_rounds), dtype=np.int32)
    patch_radii = np.zeros((num_fingers, num_rounds), dtype=np.float32)

    for finger_index in range(num_fingers):
        center = centers[finger_index]
        for round_index in range(num_rounds):
            patch_selector = (finger_ids == finger_index) & (round_ids == round_index)
            raw_patch_points = touch_points[patch_selector]
            if len(raw_patch_points) == 0:
                patch_points[finger_index, round_index] = center[None, :]
                continue
            patch_mask[finger_index, round_index] = True
            patch_counts[finger_index, round_index] = int(len(raw_patch_points))
            patch_points[finger_index, round_index] = subsample_or_repeat(
                raw_patch_points,
                points_per_patch,
                rng,
            )
            distances = np.linalg.norm(raw_patch_points - center[None, :], axis=-1)
            if distances.size > 0:
                patch_radii[finger_index, round_index] = float(np.quantile(distances, 0.95))

    patch_radii = fill_missing_radii(patch_radii, patch_mask)
    finger_mask = patch_mask.any(axis=1)
    return patch_points, patch_mask, patch_counts, patch_radii, centers, finger_mask


def build_structured_touch_payload(
    touch_data: dict[str, np.ndarray],
    points_per_patch: int,
    seed: int,
) -> dict[str, np.ndarray]:
    touch_points_all = np.asarray(touch_data["touch_points"], dtype=np.float32)
    touch_round_ids_all = np.asarray(touch_data["touch_round_ids"], dtype=np.int32)
    touch_finger_ids_all = np.asarray(touch_data["touch_finger_ids"], dtype=np.int32)
    touch_centers_all = np.asarray(touch_data["touch_centers"], dtype=np.float32)

    if touch_points_all.ndim != 3:
        raise ValueError(f"Expected touch_points with shape [V, N, 3], got {touch_points_all.shape}.")

    structured_points = []
    structured_masks = []
    structured_counts = []
    structured_radii = []
    structured_centers = []
    structured_finger_masks = []

    num_views = int(touch_points_all.shape[0])
    for view_index in range(num_views):
        rng = np.random.default_rng(int(seed) + 1009 * (view_index + 1))
        (
            patch_points,
            patch_mask,
            patch_counts,
            patch_radii,
            patch_centers,
            finger_mask,
        ) = structure_single_view(
            touch_points=touch_points_all[view_index],
            round_ids=touch_round_ids_all[view_index],
            finger_ids=touch_finger_ids_all[view_index],
            touch_centers=touch_centers_all[view_index],
            points_per_patch=points_per_patch,
            rng=rng,
        )
        structured_points.append(patch_points)
        structured_masks.append(patch_mask)
        structured_counts.append(patch_counts)
        structured_radii.append(patch_radii)
        structured_centers.append(patch_centers)
        structured_finger_masks.append(finger_mask)

    return {
        "patch_points": np.stack(structured_points, axis=0).astype(np.float32),
        "patch_mask": np.stack(structured_masks, axis=0).astype(bool),
        "patch_counts": np.stack(structured_counts, axis=0).astype(np.int32),
        "patch_radii": np.stack(structured_radii, axis=0).astype(np.float32),
        "patch_centers": np.stack(structured_centers, axis=0).astype(np.float32),
        "finger_mask": np.stack(structured_finger_masks, axis=0).astype(bool),
        "touch_view_indices": np.arange(num_views, dtype=np.int32),
        "points_per_patch": np.asarray(points_per_patch, dtype=np.int32),
        "num_touch_views": np.asarray(num_views, dtype=np.int32),
    }


def process_single_object(job: dict) -> dict:
    from SDF_batch_sampling_new_paper_idea import (
        build_raycast_scene,
        compute_query_sdf_with_raycasting,
        sample_query_points_near_surface,
        sample_surface_points_for_storage,
    )
    from SDF_batch_sampling_new_paper_idea_shapenetcore_all_10touch_mujoco_coverage_onefolder import (
        generate_mujoco_touch_data_coverage_aware,
        load_tactistruct_pipeline_module,
    )

    obj_path = str(job["obj_path"])
    out_path = Path(job["out_path"])
    asset_path = Path(job["asset_path"])

    if out_path.exists() and not bool(job["overwrite"]):
        return {
            "status": "skipped",
            "obj_path": obj_path,
            "out_path": str(out_path),
            "message": "exists",
        }

    try:
        pipeline = load_tactistruct_pipeline_module()
        seed = object_seed(job["base_seed"], obj_path)
        mesh_name = Path(obj_path).stem

        source_mesh = pipeline.load_input_mesh(Path(obj_path))
        normalized_mesh, transform = pipeline.normalize_mesh(
            source_mesh,
            float(job["normalization_bound"]),
        )

        asset_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_mesh.export(asset_path)

        surface_points, surface_normals = sample_surface_points_for_storage(
            normalized_mesh,
            num_surface_points=int(job["num_surface_points"]),
        )
        scene = build_raycast_scene(normalized_mesh)
        query_points = sample_query_points_near_surface(
            surface_points=surface_points,
            number_of_points=int(job["num_query_points"]),
        )
        query_sdf = compute_query_sdf_with_raycasting(
            scene=scene,
            query_points=query_points,
            mesh_is_watertight=normalized_mesh.is_watertight,
            surface_points=surface_points,
            surface_normals=surface_normals,
            occupancy_nsamples=11,
            near_surface_sign_band=0.01,
        )

        touch_data = generate_mujoco_touch_data_coverage_aware(
            pipeline=pipeline,
            normalized_mesh=normalized_mesh,
            normalized_mesh_path=asset_path,
            num_tactile_samples=int(job["num_tactile_samples"]),
            tactile_num_fingers=int(job["tactile_num_fingers"]),
            tactile_points_per_finger=int(job["tactile_points_per_finger"]),
            dense_surface_sample_n=int(job["dense_surface_sample_n"]),
            candidate_touch_samples=int(job["candidate_touch_samples"]),
            patch_radius_ratio=float(job["tactile_patch_radius_ratio"]),
            min_touch_separation_ratio=float(job["tactile_min_touch_separation_ratio"]),
            touch_mode="sphere",
            probe_geom="sphere",
            probe_radius=0.05,
            probe_capsule_half_length=0.04,
            probe_box_half_extents=np.asarray([0.03, 0.03, 0.04], dtype=np.float32),
            approach_offset=0.18,
            indentation_depth=0.01,
            approach_steps=80,
            background_color=np.asarray([0.88, 0.94, 1.0], dtype=np.float32),
            seed=seed,
        )

        structured_touch = build_structured_touch_payload(
            touch_data=touch_data,
            points_per_patch=int(job["structured_points_per_patch"]),
            seed=seed,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            surface_points=surface_points.astype(np.float32),
            surface_normals=surface_normals.astype(np.float32),
            query_points=query_points.astype(np.float32),
            query_sdf=query_sdf.astype(np.float32),
            patch_points=structured_touch["patch_points"],
            patch_mask=structured_touch["patch_mask"],
            patch_counts=structured_touch["patch_counts"],
            patch_radii=structured_touch["patch_radii"],
            patch_centers=structured_touch["patch_centers"],
            finger_mask=structured_touch["finger_mask"],
            touch_view_indices=structured_touch["touch_view_indices"],
            points_per_patch=structured_touch["points_per_patch"],
            num_touch_views=structured_touch["num_touch_views"],
            planning_surface_coverage_ratio=touch_data["planning_surface_coverage_ratio"],
            planning_view_coverage_ratio=touch_data["planning_view_coverage_ratio"],
            touch_center_normals=touch_data["touch_center_normals"],
            touch_target_points=touch_data["touch_target_points"],
            touch_target_normals=touch_data["touch_target_normals"],
            touch_probe_positions=touch_data["touch_probe_positions"],
            touch_probe_quaternions_wxyz=touch_data["touch_probe_quaternions_wxyz"],
            touch_patch_source_counts=touch_data["touch_patch_source_counts"],
            touch_coverage_progress=touch_data["touch_coverage_progress"],
            object_center=transform.center.astype(np.float32),
            object_scale=np.asarray(transform.scale, dtype=np.float32),
            normalization_bound=np.asarray(transform.target_bound, dtype=np.float32),
            source_mesh=np.asarray(obj_path),
            normalized_mesh_asset=np.asarray(str(asset_path)),
            mesh_name=np.asarray(mesh_name),
            num_tactile_samples=np.asarray(job["num_tactile_samples"], dtype=np.int32),
            tactile_num_fingers=np.asarray(job["tactile_num_fingers"], dtype=np.int32),
        )

        return {
            "status": "ok",
            "obj_path": obj_path,
            "out_path": str(out_path),
            "coverage": float(touch_data["planning_surface_coverage_ratio"]),
            "num_touch_views": int(structured_touch["num_touch_views"]),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "obj_path": obj_path,
            "out_path": str(out_path),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def build_jobs(args: argparse.Namespace) -> list[dict]:
    from SDF_batch_sampling_new_paper_idea_shapenetcore_all import (
        find_shapenet_obj_files,
        iter_category_dirs,
    )

    root_dir = Path(args.root_dir).resolve()
    category_names = parse_category_names(args.category_names)
    category_dirs = list(iter_category_dirs(str(root_dir), category_names=category_names))

    jobs = []
    for category_dir in category_dirs:
        obj_paths = find_shapenet_obj_files(
            category_dir,
            max_objects=args.max_objects_per_category,
        )
        for obj_path in obj_paths:
            out_path = build_flat_output_path(
                obj_path=obj_path,
                root_dir=str(root_dir),
                output_folder_name=args.output_folder_name,
            )
            asset_path = build_asset_export_path(
                obj_path=obj_path,
                root_dir=str(root_dir),
                asset_folder_name=args.asset_folder_name,
            )
            jobs.append(
                {
                    "obj_path": obj_path,
                    "out_path": out_path,
                    "asset_path": asset_path,
                    "overwrite": bool(args.overwrite),
                    "base_seed": int(args.base_seed),
                    "normalization_bound": float(args.normalization_bound),
                    "num_surface_points": int(args.num_surface_points),
                    "num_query_points": int(args.num_query_points),
                    "num_tactile_samples": int(args.num_tactile_samples),
                    "tactile_num_fingers": int(args.tactile_num_fingers),
                    "tactile_points_per_finger": int(args.tactile_points_per_finger),
                    "structured_points_per_patch": int(args.structured_points_per_patch),
                    "dense_surface_sample_n": int(args.dense_surface_sample_n),
                    "candidate_touch_samples": int(args.candidate_touch_samples),
                    "tactile_patch_radius_ratio": float(args.tactile_patch_radius_ratio),
                    "tactile_min_touch_separation_ratio": float(args.tactile_min_touch_separation_ratio),
                }
            )
    return jobs


def run_parallel_jobs(args: argparse.Namespace, jobs: list[dict]) -> list[dict]:
    if not jobs:
        return []

    max_workers = max(1, int(args.max_workers))
    if max_workers == 1:
        return [process_single_object(job) for job in jobs]

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {executor.submit(process_single_object, job): job for job in jobs}
        for future in concurrent.futures.as_completed(future_to_job):
            result = future.result()
            results.append(result)
            if result["status"] == "failed" and bool(args.fail_fast):
                for pending in future_to_job:
                    pending.cancel()
                break
    return results


def summarise_results(results: list[dict]) -> dict:
    summary = {
        "ok": 0,
        "skipped": 0,
        "failed": 0,
        "mean_coverage": 0.0,
        "outputs": [],
        "failures": [],
    }
    coverages = []
    for result in results:
        status = result["status"]
        summary[status] += 1
        if status == "ok":
            summary["outputs"].append(result["out_path"])
            coverages.append(float(result.get("coverage", 0.0)))
        elif status == "failed":
            summary["failures"].append(
                {
                    "obj_path": result["obj_path"],
                    "error": result.get("error"),
                    "traceback": result.get("traceback"),
                }
            )
    if coverages:
        summary["mean_coverage"] = float(np.mean(coverages))
    return summary


def main() -> None:
    args = parse_args()
    jobs = build_jobs(args)
    print(f"[INFO] prepared {len(jobs)} preprocessing jobs")
    print(f"[INFO] max_workers={int(args.max_workers)} output_folder={args.output_folder_name}")

    results = run_parallel_jobs(args, jobs)
    for result in results:
        status = result["status"].upper()
        if result["status"] == "ok":
            print(
                f"[{status}] {result['obj_path']} -> {result['out_path']} "
                f"(coverage={result['coverage']:.4f})"
            )
        elif result["status"] == "skipped":
            print(f"[{status}] {result['obj_path']} ({result['message']})")
        else:
            print(f"[{status}] {result['obj_path']}")
            print(result.get("error", "unknown error"))

    summary = summarise_results(results)
    summary_path = (
        Path(args.root_dir).resolve()
        / args.output_folder_name
        / "preprocess_summary.json"
    )
    save_json(summary_path, summary)
    print(f"[SUMMARY] saved to {summary_path}")
    print(
        f"[SUMMARY] ok={summary['ok']} skipped={summary['skipped']} "
        f"failed={summary['failed']} mean_coverage={summary['mean_coverage']:.4f}"
    )

    if summary["failed"] > 0 and bool(args.fail_fast):
        raise RuntimeError("At least one preprocessing job failed in fail-fast mode.")


if __name__ == "__main__":
    main()
