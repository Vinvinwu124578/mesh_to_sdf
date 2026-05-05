from __future__ import annotations

import argparse
import concurrent.futures
import os
import traceback
from pathlib import Path

import numpy as np

from adaptive_patch import (
    build_asset_export_path,
    build_flat_output_path,
    build_structured_touch_payload,
    object_seed,
)
from common import DEFAULT_SHAPENET_ROOT, save_json


DEFAULT_OUTPUT_FOLDER = "tactistruct_progressive_attn_fix_adaptive_confidence_no_pose_onefolder"
DEFAULT_ASSET_FOLDER = "tactistruct_progressive_attn_fix_adaptive_confidence_no_pose_assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run coverage-aware MuJoCo tactile preprocessing in parallel, then convert tactile patches into "
            "an adaptive-confidence representation for progressive_attn_fix-style training."
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
    parser.add_argument("--adaptive-minor-radius-floor-ratio", type=float, default=0.30)
    parser.add_argument("--adaptive-minor-radius-scale", type=float, default=1.15)
    parser.add_argument("--adaptive-major-radius-scale", type=float, default=1.05)
    parser.add_argument("--adaptive-plane-gap-ratio", type=float, default=0.28)
    parser.add_argument("--adaptive-min-points-per-patch", type=int, default=24)
    parser.add_argument("--confidence-floor", type=float, default=0.08)
    return parser.parse_args()


def parse_category_names(value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [item.strip() for item in value.split(",")]
    names = [item for item in names if item]
    return names or None


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
        return {"status": "skipped", "obj_path": obj_path, "out_path": str(out_path), "message": "exists"}

    try:
        pipeline = load_tactistruct_pipeline_module()
        seed = object_seed(job["base_seed"], obj_path)
        mesh_name = Path(obj_path).stem

        source_mesh = pipeline.load_input_mesh(Path(obj_path))
        normalized_mesh, transform = pipeline.normalize_mesh(source_mesh, float(job["normalization_bound"]))

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
            surface_points=surface_points,
            surface_normals=surface_normals,
            points_per_patch=int(job["structured_points_per_patch"]),
            seed=seed,
            adaptive_minor_radius_floor_ratio=float(job["adaptive_minor_radius_floor_ratio"]),
            adaptive_minor_radius_scale=float(job["adaptive_minor_radius_scale"]),
            adaptive_major_radius_scale=float(job["adaptive_major_radius_scale"]),
            adaptive_plane_gap_ratio=float(job["adaptive_plane_gap_ratio"]),
            adaptive_min_points_per_patch=int(job["adaptive_min_points_per_patch"]),
            confidence_floor=float(job["confidence_floor"]),
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            surface_points=surface_points.astype(np.float32),
            surface_normals=surface_normals.astype(np.float32),
            query_points=query_points.astype(np.float32),
            query_sdf=query_sdf.astype(np.float32),
            patch_points=structured_touch["patch_points"],
            patch_point_normals=structured_touch["patch_point_normals"],
            patch_point_confidence=structured_touch["patch_point_confidence"],
            patch_point_band_width=structured_touch["patch_point_band_width"],
            patch_mask=structured_touch["patch_mask"],
            patch_counts=structured_touch["patch_counts"],
            patch_radii=structured_touch["patch_radii"],
            patch_centers=structured_touch["patch_centers"],
            finger_mask=structured_touch["finger_mask"],
            patch_confidence=structured_touch["patch_confidence"],
            patch_edge_score=structured_touch["patch_edge_score"],
            patch_band_width=structured_touch["patch_band_width"],
            patch_major_extent=structured_touch["patch_major_extent"],
            patch_minor_extent=structured_touch["patch_minor_extent"],
            patch_normal_variance=structured_touch["patch_normal_variance"],
            patch_plane_split_flag=structured_touch["patch_plane_split_flag"],
            patch_reachability_margin=structured_touch["patch_reachability_margin"],
            patch_target_contact_offset=structured_touch["patch_target_contact_offset"],
            patch_target_contact_offset_ratio=structured_touch["patch_target_contact_offset_ratio"],
            patch_adaptive_major_radius=structured_touch["patch_adaptive_major_radius"],
            patch_adaptive_minor_radius=structured_touch["patch_adaptive_minor_radius"],
            touch_view_indices=structured_touch["touch_view_indices"],
            points_per_patch=structured_touch["points_per_patch"],
            num_touch_views=structured_touch["num_touch_views"],
            adaptive_minor_radius_floor_ratio=structured_touch["adaptive_minor_radius_floor_ratio"],
            adaptive_minor_radius_scale=structured_touch["adaptive_minor_radius_scale"],
            adaptive_major_radius_scale=structured_touch["adaptive_major_radius_scale"],
            adaptive_plane_gap_ratio=structured_touch["adaptive_plane_gap_ratio"],
            adaptive_min_points_per_patch=structured_touch["adaptive_min_points_per_patch"],
            confidence_floor=structured_touch["confidence_floor"],
            planning_surface_coverage_ratio=touch_data["planning_surface_coverage_ratio"],
            planning_view_coverage_ratio=touch_data["planning_view_coverage_ratio"],
            planning_reachable_surface_fraction=touch_data.get("planning_reachable_surface_fraction", np.asarray(1.0, dtype=np.float32)),
            planning_reachable_surface_point_count=touch_data.get("planning_reachable_surface_point_count", np.asarray(0, dtype=np.int32)),
            planning_dense_surface_point_count=touch_data.get("planning_dense_surface_point_count", np.asarray(0, dtype=np.int32)),
            planning_candidate_point_count=touch_data.get("planning_candidate_point_count", np.asarray(0, dtype=np.int32)),
            touch_center_normals=touch_data["touch_center_normals"],
            touch_target_points=touch_data["touch_target_points"],
            touch_target_normals=touch_data["touch_target_normals"],
            touch_contact_points=touch_data["touch_contact_points"],
            touch_contact_normals=touch_data["touch_contact_normals"],
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

        confidence_mean = (
            float(structured_touch["patch_confidence"][structured_touch["patch_mask"]].mean())
            if np.any(structured_touch["patch_mask"])
            else 0.0
        )
        return {
            "status": "ok",
            "obj_path": obj_path,
            "out_path": str(out_path),
            "coverage": float(touch_data["planning_surface_coverage_ratio"]),
            "confidence_mean": confidence_mean,
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
    from SDF_batch_sampling_new_paper_idea_shapenetcore_all import find_shapenet_obj_files, iter_category_dirs

    root_dir = Path(args.root_dir).resolve()
    category_names = parse_category_names(args.category_names)
    category_dirs = list(iter_category_dirs(str(root_dir), category_names=category_names))

    jobs = []
    for category_dir in category_dirs:
        obj_paths = find_shapenet_obj_files(category_dir, max_objects=args.max_objects_per_category)
        for obj_path in obj_paths:
            jobs.append(
                {
                    "obj_path": obj_path,
                    "out_path": build_flat_output_path(obj_path, str(root_dir), args.output_folder_name),
                    "asset_path": build_asset_export_path(obj_path, str(root_dir), args.asset_folder_name),
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
                    "adaptive_minor_radius_floor_ratio": float(args.adaptive_minor_radius_floor_ratio),
                    "adaptive_minor_radius_scale": float(args.adaptive_minor_radius_scale),
                    "adaptive_major_radius_scale": float(args.adaptive_major_radius_scale),
                    "adaptive_plane_gap_ratio": float(args.adaptive_plane_gap_ratio),
                    "adaptive_min_points_per_patch": int(args.adaptive_min_points_per_patch),
                    "confidence_floor": float(args.confidence_floor),
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
    summary = {"ok": 0, "skipped": 0, "failed": 0, "mean_coverage": 0.0, "mean_confidence": 0.0, "outputs": [], "failures": []}
    coverages = []
    confidences = []
    for result in results:
        summary[result["status"]] += 1
        if result["status"] == "ok":
            summary["outputs"].append(result["out_path"])
            coverages.append(float(result.get("coverage", 0.0)))
            confidences.append(float(result.get("confidence_mean", 0.0)))
        elif result["status"] == "failed":
            summary["failures"].append(
                {"obj_path": result["obj_path"], "error": result.get("error"), "traceback": result.get("traceback")}
            )
    if coverages:
        summary["mean_coverage"] = float(np.mean(coverages))
    if confidences:
        summary["mean_confidence"] = float(np.mean(confidences))
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
            print(f"[{status}] {result['obj_path']} -> {result['out_path']} (coverage={result['coverage']:.4f}, conf={result['confidence_mean']:.4f})")
        elif result["status"] == "skipped":
            print(f"[{status}] {result['obj_path']} ({result['message']})")
        else:
            print(f"[{status}] {result['obj_path']}")
            print(result.get("error", "unknown error"))

    summary = summarise_results(results)
    summary_path = Path(args.root_dir).resolve() / args.output_folder_name / "preprocess_summary.json"
    save_json(summary_path, summary)
    print(f"[SUMMARY] saved to {summary_path}")
    print(f"[SUMMARY] ok={summary['ok']} skipped={summary['skipped']} failed={summary['failed']} mean_coverage={summary['mean_coverage']:.4f} mean_confidence={summary['mean_confidence']:.4f}")
    if summary["failed"] > 0 and bool(args.fail_fast):
        raise RuntimeError("At least one preprocessing job failed in fail-fast mode.")


if __name__ == "__main__":
    main()
