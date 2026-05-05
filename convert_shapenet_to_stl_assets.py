import argparse
import concurrent.futures
import json
import os
import traceback
from pathlib import Path

import numpy as np

from SDF_batch_sampling_new_paper_idea import sample_surface_points_for_storage
from SDF_batch_sampling_new_paper_idea_shapenetcore_all import (
    find_shapenet_obj_files,
    iter_category_dirs,
)
from SDF_batch_sampling_new_paper_idea_shapenetcore_all_10touch_mujoco_coverage_onefolder_paired_watertight import (
    build_sign_proxy_mesh,
    load_tactistruct_pipeline_module,
    object_seed,
    orient_normals_outward_from_center,
    simplify_mesh_for_mujoco,
)


DEFAULT_MANIFOLDPLUS_PATH = (
    "wsl:/mnt/c/Users/wudaw/Downloads/mesh_to_sdf-master/mesh_to_sdf-master/"
    "mesh_to_sdf/external_tools/ManifoldPlus/build_conda_path/manifold"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a single OBJ file or ShapeNetCore model_normalized.obj files "
            "to normalized STL assets. Optionally builds a watertight mesh first."
        )
    )
    parser.add_argument("--input-obj", type=str, default=None)
    parser.add_argument("--output-stl", type=str, default=None)
    parser.add_argument(
        "--root-dir",
        type=str,
        default=r"C:\Users\wudaw\Downloads\ShapeNetCore\ShapeNetCore",
    )
    parser.add_argument("--category-names", type=str, default=None)
    parser.add_argument("--max-objects-per-category", type=int, default=275)
    parser.add_argument(
        "--output-folder-name",
        type=str,
        default="shapenet_stl_assets_full_watertight_manifoldplus",
    )
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--normalization-bound", type=float, default=0.9)
    parser.add_argument("--num-surface-points", type=int, default=120000)
    parser.add_argument(
        "--watertight-proxy-mode",
        type=str,
        default="manifoldplus",
        choices=("repair", "poisson", "pymeshlab_poisson", "pymeshfix", "manifoldplus", "convex_hull", "none"),
    )
    parser.add_argument(
        "--non-watertight-policy",
        type=str,
        default="skip",
        choices=("skip", "continue"),
    )
    parser.add_argument("--proxy-poisson-samples", type=int, default=120000)
    parser.add_argument("--proxy-poisson-depth", type=int, default=8)
    parser.add_argument("--proxy-poisson-full-depth", type=int, default=5)
    parser.add_argument("--proxy-poisson-threads", type=int, default=8)
    parser.add_argument("--manifoldplus-path", type=str, default=DEFAULT_MANIFOLDPLUS_PATH)
    parser.add_argument("--manifoldplus-depth", type=int, default=8)
    parser.add_argument(
        "--mujoco-max-faces",
        type=int,
        default=190000,
        help="Decimate exported STL to stay below MuJoCo's 200000 face STL limit.",
    )
    parser.add_argument("--base-seed", type=int, default=42)
    return parser.parse_args()


def parse_category_names(value):
    if value is None:
        return None
    names = [item.strip() for item in str(value).split(",")]
    names = [item for item in names if item]
    return names or None


def safe_asset_name(obj_path, root_dir):
    rel_path = os.path.relpath(obj_path, root_dir)
    rel_without_ext = os.path.splitext(rel_path)[0]
    return rel_without_ext.replace("\\", "__").replace("/", "__").replace(":", "_")


def build_stl_output_path(obj_path, root_dir, output_folder_name):
    output_dir = Path(root_dir) / output_folder_name
    return output_dir / f"{safe_asset_name(obj_path, root_dir)}__normalized.stl"


def build_single_stl_output_path(obj_path, output_stl=None):
    obj_path = Path(obj_path).resolve()
    if output_stl is not None and str(output_stl).strip():
        return Path(output_stl).resolve()
    return obj_path.with_name(f"{obj_path.stem}__normalized.stl")


def convert_single_obj_to_stl(job):
    obj_path = Path(job["obj_path"])
    out_path = Path(job["out_path"])
    if out_path.exists() and not bool(job["overwrite"]):
        return {
            "status": "skipped",
            "obj_path": str(obj_path),
            "out_path": str(out_path),
            "message": "exists",
        }

    pipeline = load_tactistruct_pipeline_module()
    source_mesh = pipeline.load_input_mesh(obj_path)
    normalized_mesh, transform = pipeline.normalize_mesh(
        source_mesh,
        float(job["normalization_bound"]),
    )

    mesh_source = "original_normalized"
    if str(job["watertight_proxy_mode"]).strip().lower() != "none":
        surface_points, surface_normals = sample_surface_points_for_storage(
            normalized_mesh,
            num_surface_points=int(job["num_surface_points"]),
        )
        surface_normals = orient_normals_outward_from_center(
            surface_points,
            surface_normals,
            normalized_mesh.bounding_box.centroid,
        )
        watertight_mesh, watertight_source = build_sign_proxy_mesh(
            mesh=normalized_mesh,
            surface_points=surface_points,
            surface_normals=surface_normals,
            mode=job["watertight_proxy_mode"],
            poisson_sample_count=int(job["proxy_poisson_samples"]),
            poisson_depth=int(job["proxy_poisson_depth"]),
            poisson_full_depth=int(job["proxy_poisson_full_depth"]),
            poisson_threads=int(job["proxy_poisson_threads"]),
            manifoldplus_path=job["manifoldplus_path"],
            manifoldplus_depth=int(job["manifoldplus_depth"]),
            seed=int(job["seed"]) + 104729,
        )
        if watertight_mesh is not None and watertight_mesh.is_watertight:
            normalized_mesh = watertight_mesh
            mesh_source = f"watertight_{watertight_source}"
        elif str(job["non_watertight_policy"]).strip().lower() == "skip":
            raise RuntimeError("No watertight mesh could be built.")
        else:
            mesh_source = f"non_watertight_fallback_{watertight_source}"

    export_mesh, was_simplified = simplify_mesh_for_mujoco(
        normalized_mesh,
        max_faces=int(job["mujoco_max_faces"]),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_mesh.export(out_path)

    metadata = {
        "status": "ok",
        "obj_path": str(obj_path),
        "out_path": str(out_path),
        "mesh_source": str(mesh_source),
        "mesh_is_watertight": bool(normalized_mesh.is_watertight),
        "export_faces": int(len(export_mesh.faces)),
        "export_vertices": int(len(export_mesh.vertices)),
        "mujoco_mesh_was_simplified": bool(was_simplified),
        "object_center": np.asarray(transform.center, dtype=float).tolist(),
        "object_scale": float(transform.scale),
        "normalization_bound": float(transform.target_bound),
    }
    return metadata


def build_jobs(args):
    if args.input_obj is not None and str(args.input_obj).strip():
        input_obj = Path(args.input_obj).resolve()
        output_path = build_single_stl_output_path(
            input_obj,
            output_stl=args.output_stl,
        )
        root_dir = input_obj.parent
        jobs = [
            {
                "obj_path": str(input_obj),
                "out_path": str(output_path),
                "overwrite": bool(args.overwrite),
                "normalization_bound": float(args.normalization_bound),
                "num_surface_points": int(args.num_surface_points),
                "watertight_proxy_mode": str(args.watertight_proxy_mode),
                "non_watertight_policy": str(args.non_watertight_policy),
                "proxy_poisson_samples": int(args.proxy_poisson_samples),
                "proxy_poisson_depth": int(args.proxy_poisson_depth),
                "proxy_poisson_full_depth": int(args.proxy_poisson_full_depth),
                "proxy_poisson_threads": int(args.proxy_poisson_threads),
                "manifoldplus_path": str(args.manifoldplus_path),
                "manifoldplus_depth": int(args.manifoldplus_depth),
                "mujoco_max_faces": int(args.mujoco_max_faces),
                "seed": int(object_seed(args.base_seed, input_obj)),
            }
        ]
        return root_dir, [], jobs, output_path.parent

    root_dir = Path(args.root_dir).resolve()
    category_dirs = list(
        iter_category_dirs(str(root_dir), category_names=parse_category_names(args.category_names))
    )
    jobs = []
    for category_dir in category_dirs:
        obj_paths = find_shapenet_obj_files(
            category_dir,
            max_objects=args.max_objects_per_category,
        )
        for obj_path in obj_paths:
            jobs.append(
                {
                    "obj_path": str(obj_path),
                    "out_path": str(build_stl_output_path(obj_path, root_dir, args.output_folder_name)),
                    "overwrite": bool(args.overwrite),
                    "normalization_bound": float(args.normalization_bound),
                    "num_surface_points": int(args.num_surface_points),
                    "watertight_proxy_mode": str(args.watertight_proxy_mode),
                    "non_watertight_policy": str(args.non_watertight_policy),
                    "proxy_poisson_samples": int(args.proxy_poisson_samples),
                    "proxy_poisson_depth": int(args.proxy_poisson_depth),
                    "proxy_poisson_full_depth": int(args.proxy_poisson_full_depth),
                    "proxy_poisson_threads": int(args.proxy_poisson_threads),
                    "manifoldplus_path": str(args.manifoldplus_path),
                    "manifoldplus_depth": int(args.manifoldplus_depth),
                    "mujoco_max_faces": int(args.mujoco_max_faces),
                    "seed": int(object_seed(args.base_seed, obj_path)),
                }
            )
    return root_dir, category_dirs, jobs, root_dir / args.output_folder_name


def run_jobs(jobs, max_workers=1, fail_fast=False):
    max_workers = max(1, int(max_workers))
    if max_workers == 1:
        results = []
        for job in jobs:
            try:
                results.append(convert_single_obj_to_stl(job))
            except Exception as exc:
                result = {
                    "status": "failed",
                    "obj_path": str(job["obj_path"]),
                    "out_path": str(job["out_path"]),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                results.append(result)
                if fail_fast:
                    break
        return results

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {executor.submit(convert_single_obj_to_stl, job): job for job in jobs}
        for future in concurrent.futures.as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "status": "failed",
                    "obj_path": str(job["obj_path"]),
                    "out_path": str(job["out_path"]),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            results.append(result)
            status = result["status"].upper()
            if result["status"] == "ok":
                print(
                    f"[{status}] {result['out_path']} "
                    f"(faces={result['export_faces']}, watertight={result['mesh_is_watertight']})"
                )
            elif result["status"] == "skipped":
                print(f"[{status}] {result['out_path']} ({result['message']})")
            else:
                print(f"[{status}] {result['obj_path']}: {result.get('error', 'unknown error')}")
                if fail_fast:
                    for pending in future_to_job:
                        pending.cancel()
                    break
    return results


def main():
    args = parse_args()
    root_dir, category_dirs, jobs, output_dir = build_jobs(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_obj is not None and str(args.input_obj).strip():
        print(f"[INFO] single input OBJ: {Path(args.input_obj).resolve()}")
    else:
        print(f"[INFO] found {len(category_dirs)} category folders under root.")
    print(f"[INFO] prepared {len(jobs)} STL conversion jobs")
    print(f"[INFO] output dir: {output_dir}")
    print(f"[INFO] max_workers={int(args.max_workers)} overwrite={bool(args.overwrite)}")

    results = run_jobs(
        jobs=jobs,
        max_workers=args.max_workers,
        fail_fast=bool(args.fail_fast),
    )

    summary = {
        "ok": int(sum(1 for item in results if item["status"] == "ok")),
        "skipped": int(sum(1 for item in results if item["status"] == "skipped")),
        "failed": int(sum(1 for item in results if item["status"] == "failed")),
        "results": results,
    }
    summary_path = output_dir / "stl_conversion_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(
        f"[SUMMARY] ok={summary['ok']} skipped={summary['skipped']} "
        f"failed={summary['failed']}"
    )
    print(f"[SUMMARY] saved to {summary_path}")


if __name__ == "__main__":
    main()
