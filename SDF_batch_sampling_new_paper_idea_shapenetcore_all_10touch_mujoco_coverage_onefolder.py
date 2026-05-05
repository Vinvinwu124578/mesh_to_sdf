import argparse
import concurrent.futures
import hashlib
import importlib.util
import multiprocessing
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from SDF_batch_sampling_new_paper_idea import (
    build_raycast_scene,
    compute_query_sdf_with_raycasting,
    sample_query_points_near_surface,
    sample_surface_points_for_storage,
)
from SDF_batch_sampling_new_paper_idea_shapenetcore_all import (
    find_shapenet_obj_files,
    iter_category_dirs,
)


TACTISTRUCT_PIPELINE_PATH = Path(
    r"C:\Users\wudaw\Downloads\Tactistruct-main\Tactistruct-main\pipeline_tactile_visualisation.py"
)
_PIPELINE_MODULE = None


def load_tactistruct_pipeline_module():
    global _PIPELINE_MODULE
    if _PIPELINE_MODULE is not None:
        return _PIPELINE_MODULE

    if not TACTISTRUCT_PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Tactistruct pipeline file not found: {TACTISTRUCT_PIPELINE_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "tactistruct_pipeline_tactile_visualisation",
        str(TACTISTRUCT_PIPELINE_PATH),
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not create import spec for: {TACTISTRUCT_PIPELINE_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _PIPELINE_MODULE = module
    return module


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run coverage-aware MuJoCo tactile preprocessing for ShapeNetCore and save "
            "flat onefolder NPZ outputs. Uses process-based parallelism for stability."
        )
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=r"C:/Users/wudaw/Downloads/ShapeNetCore/ShapeNetCore",
    )
    parser.add_argument("--category-names", type=str, default=None)
    parser.add_argument("--max-objects-per-category", type=int, default=275)
    parser.add_argument(
        "--output-folder-name",
        type=str,
        default="tactistruct_npz_shapenet_mujoco_coverage_onefolder",
    )
    parser.add_argument(
        "--asset-folder-name",
        type=str,
        default="tactistruct_npz_shapenet_mujoco_coverage_assets",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 1))),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--num-tactile-samples", type=int, default=10)
    parser.add_argument("--tactile-num-fingers", type=int, default=10)
    parser.add_argument("--tactile-points-per-finger", type=int, default=3000)
    parser.add_argument("--dense-surface-sample-n", type=int, default=120000)
    parser.add_argument("--candidate-touch-samples", type=int, default=6000)
    parser.add_argument("--tactile-patch-radius-ratio", type=float, default=0.10)
    parser.add_argument("--tactile-min-touch-separation-ratio", type=float, default=0.055)
    parser.add_argument("--tactile-patch-thickness-ratio", type=float, default=0.035)
    parser.add_argument("--patch-min-normal-cos", type=float, default=0.05)
    parser.add_argument("--tactile-patch-dominant-normal-gap-cos", type=float, default=0.18)
    parser.add_argument("--tactile-patch-plane-gap-ratio", type=float, default=0.35)
    parser.add_argument("--tactile-patch-link-radius-ratio", type=float, default=0.0)
    parser.add_argument("--max-target-contact-offset-ratio", type=float, default=0.60)
    parser.add_argument("--tactile-reachable-clearance-ratio", type=float, default=0.92)
    parser.add_argument("--tactile-reachable-approach-steps", type=int, default=5)
    parser.add_argument("--normalization-bound", type=float, default=0.9)
    parser.add_argument("--num-surface-points", type=int, default=235000)
    parser.add_argument("--num-query-points", type=int, default=250000)
    parser.add_argument(
        "--query-uniform-region",
        type=str,
        default="cube",
        choices=("sphere", "cube"),
        help="Global random query support region mixed with near-surface samples. "
        "Use 'cube' to match inference over [-1, 1]^3.",
    )
    parser.add_argument("--base-seed", type=int, default=42)
    return parser.parse_args()


def parse_category_names(value):
    if value is None:
        return None
    names = [item.strip() for item in str(value).split(",")]
    names = [item for item in names if item]
    return names or None


def object_seed(base_seed, obj_path):
    digest = hashlib.sha1(str(obj_path).encode("utf-8")).hexdigest()[:8]
    return int(base_seed) + int(digest, 16)


def build_flat_output_path(obj_path, root_dir, output_folder_name):
    rel_path = os.path.relpath(obj_path, root_dir)
    rel_without_ext = os.path.splitext(rel_path)[0]
    safe_name = rel_without_ext.replace("\\", "__").replace("/", "__").replace(":", "_")
    out_dir = os.path.join(root_dir, output_folder_name)
    return os.path.join(out_dir, safe_name + ".npz")


def build_asset_export_path(obj_path, root_dir, asset_folder_name):
    rel_path = os.path.relpath(obj_path, root_dir)
    rel_without_ext = os.path.splitext(rel_path)[0]
    safe_name = rel_without_ext.replace("\\", "__").replace("/", "__").replace(":", "_")
    asset_dir = os.path.join(root_dir, asset_folder_name)
    return os.path.join(asset_dir, safe_name + "__normalized.stl")


def compute_bbox_diag(points):
    points = np.asarray(points, dtype=np.float32)
    mn = np.min(points, axis=0)
    mx = np.max(points, axis=0)
    return float(np.linalg.norm(mx - mn))


def sample_dense_surface_points(mesh, sample_count):
    points, face_ids = trimesh.sample.sample_surface(mesh, int(sample_count))
    points = points.astype(np.float32)
    normals = mesh.face_normals[face_ids].astype(np.float32)

    centroid = mesh.bounding_box.centroid.astype(np.float32)
    outward_hint = points - centroid[None, :]
    flip_mask = np.einsum("ij,ij->i", normals, outward_hint) < 0.0
    normals[flip_mask] *= -1.0
    return points, normals


def compute_scene_distance(scene, points, batch_size=65536):
    import open3d as o3d

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    distances = np.zeros((len(points),), dtype=np.float32)

    for start in range(0, len(points), int(batch_size)):
        end = min(len(points), start + int(batch_size))
        tensor = o3d.core.Tensor(
            points[start:end],
            dtype=o3d.core.Dtype.Float32,
        )
        distances[start:end] = scene.compute_distance(tensor).numpy().astype(np.float32)

    return distances


def compute_sphere_reachability_mask(
    scene,
    surface_points,
    surface_normals,
    probe_radius,
    approach_offset,
    clearance_ratio=0.92,
    approach_steps=5,
):
    surface_points = np.asarray(surface_points, dtype=np.float32).reshape(-1, 3)
    surface_normals = np.asarray(surface_normals, dtype=np.float32).reshape(-1, 3)

    if len(surface_points) == 0:
        return np.zeros((0,), dtype=bool)

    normal_norm = np.linalg.norm(surface_normals, axis=1, keepdims=True)
    surface_normals = surface_normals / np.clip(normal_norm, 1e-8, None)

    start_offset = float(probe_radius)
    end_offset = float(probe_radius) + max(float(approach_offset), 0.0)
    step_count = max(2, int(approach_steps))
    center_offsets = np.linspace(
        start_offset,
        end_offset,
        num=step_count,
        endpoint=True,
        dtype=np.float32,
    )

    clearance_threshold = float(clearance_ratio) * float(probe_radius)
    reachable_mask = np.ones((len(surface_points),), dtype=bool)

    for center_offset in center_offsets:
        probe_centers = surface_points + surface_normals * float(center_offset)
        clearance = compute_scene_distance(scene, probe_centers)
        reachable_mask &= clearance >= clearance_threshold
        if not np.any(reachable_mask):
            break

    return reachable_mask


def normalize_vector(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-8:
        return np.zeros((3,), dtype=np.float32)
    return (vector / norm).astype(np.float32)


def build_tangent_basis(normal):
    normal = normalize_vector(normal)
    if abs(float(normal[2])) < 0.9:
        helper = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        helper = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    tangent_u = np.cross(normal, helper).astype(np.float32)
    tangent_u = normalize_vector(tangent_u)
    tangent_v = np.cross(normal, tangent_u).astype(np.float32)
    tangent_v = normalize_vector(tangent_v)
    return tangent_u, tangent_v


def compute_patch_tangent_geometry(points, center_point, center_normal):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    center_point = np.asarray(center_point, dtype=np.float32).reshape(3)
    center_normal = normalize_vector(center_normal)
    tangent_u, tangent_v = build_tangent_basis(center_normal)
    offsets = points - center_point[None, :]
    uv = np.stack(
        [
            offsets @ tangent_u,
            offsets @ tangent_v,
        ],
        axis=1,
    ).astype(np.float32)

    if len(uv) <= 1:
        principal_coords = uv.astype(np.float32)
        major_extent = float(np.max(np.abs(principal_coords[:, 0]))) if len(principal_coords) else 0.0
        minor_extent = float(np.max(np.abs(principal_coords[:, 1]))) if len(principal_coords) else 0.0
        rotation = np.eye(2, dtype=np.float32)
    else:
        cov = np.cov(uv.T).astype(np.float32)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order].astype(np.float32)
        principal_coords = (uv @ eigvecs).astype(np.float32)
        major_extent = float(np.quantile(np.abs(principal_coords[:, 0]), 0.9))
        minor_extent = float(np.quantile(np.abs(principal_coords[:, 1]), 0.9))
        rotation = eigvecs

    return {
        "uv": uv.astype(np.float32),
        "principal_coords": principal_coords.astype(np.float32),
        "major_extent": float(max(major_extent, 0.0)),
        "minor_extent": float(max(minor_extent, 0.0)),
        "rotation": rotation.astype(np.float32),
        "tangent_u": tangent_u.astype(np.float32),
        "tangent_v": tangent_v.astype(np.float32),
    }


def align_normal_to_reference(normal, reference_normal):
    normal = normalize_vector(normal)
    reference_normal = normalize_vector(reference_normal)
    if np.dot(normal, reference_normal) < 0.0:
        normal = -normal
    return normal.astype(np.float32)


def estimate_patch_ids(dense_points, dense_tree, center_point, patch_radius, nearest_fallback_k=256):
    patch_ids = dense_tree.query_ball_point(center_point, r=float(patch_radius))
    if patch_ids:
        return np.asarray(patch_ids, dtype=np.int64)

    nearest_k = min(int(nearest_fallback_k), len(dense_points))
    _, nearest_ids = dense_tree.query(center_point[None, :], k=nearest_k)
    return np.asarray(nearest_ids, dtype=np.int64).reshape(-1)


def keep_patch_component_near_center(points, center_point, link_radius):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(points) <= 1 or link_radius is None or float(link_radius) <= 0.0:
        return np.arange(len(points), dtype=np.int64)

    local_tree = cKDTree(points)
    _, anchor_local = local_tree.query(center_point[None, :], k=1)
    anchor_local = int(np.asarray(anchor_local).reshape(-1)[0])

    visited = np.zeros((len(points),), dtype=bool)
    queue = [anchor_local]
    visited[anchor_local] = True

    while queue:
        current = queue.pop()
        neighbor_ids = local_tree.query_ball_point(points[current], r=float(link_radius))
        for neighbor_id in neighbor_ids:
            neighbor_id = int(neighbor_id)
            if not visited[neighbor_id]:
                visited[neighbor_id] = True
                queue.append(neighbor_id)

    return np.flatnonzero(visited).astype(np.int64)


def keep_patch_plane_cluster_near_center(signed_depth, gap_threshold):
    signed_depth = np.asarray(signed_depth, dtype=np.float32).reshape(-1)
    if len(signed_depth) <= 1 or gap_threshold is None or float(gap_threshold) <= 0.0:
        return np.arange(len(signed_depth), dtype=np.int64)

    sort_ids = np.argsort(signed_depth)
    sorted_depth = signed_depth[sort_ids]
    anchor_sorted_idx = int(np.argmin(np.abs(sorted_depth)))

    left = anchor_sorted_idx
    while left > 0:
        if float(sorted_depth[left] - sorted_depth[left - 1]) > float(gap_threshold):
            break
        left -= 1

    right = anchor_sorted_idx
    while right < len(sorted_depth) - 1:
        if float(sorted_depth[right + 1] - sorted_depth[right]) > float(gap_threshold):
            break
        right += 1

    return np.sort(sort_ids[left : right + 1].astype(np.int64))


def keep_patch_dominant_normal_cluster(
    normal_cos,
    gap_threshold,
    min_prefix_ratio=0.25,
    min_prefix_count=6,
):
    normal_cos = np.asarray(normal_cos, dtype=np.float32).reshape(-1)
    if len(normal_cos) <= 1 or gap_threshold is None or float(gap_threshold) <= 0.0:
        return np.arange(len(normal_cos), dtype=np.int64)

    sort_ids = np.argsort(-normal_cos)
    sorted_cos = normal_cos[sort_ids]
    gaps = sorted_cos[:-1] - sorted_cos[1:]
    if len(gaps) == 0:
        return np.arange(len(normal_cos), dtype=np.int64)

    min_prefix = max(
        int(min_prefix_count),
        int(np.ceil(float(min_prefix_ratio) * float(len(sorted_cos)))),
    )
    min_prefix = min(max(2, min_prefix), len(sorted_cos) - 1)

    split_candidates = np.flatnonzero(gaps > float(gap_threshold))
    split_candidates = split_candidates[split_candidates + 1 >= min_prefix]
    if len(split_candidates) == 0:
        return np.arange(len(normal_cos), dtype=np.int64)

    split_idx = int(split_candidates[0])
    return np.sort(sort_ids[: split_idx + 1].astype(np.int64))


def filter_patch_ids(
    dense_points,
    dense_normals,
    dense_tree,
    center_point,
    center_normal,
    patch_radius,
    patch_thickness,
    min_normal_cos=0.05,
    patch_dominant_normal_gap_cos=0.18,
    patch_plane_gap_ratio=0.35,
    patch_link_radius_ratio=0.0,
    nearest_fallback_k=256,
):
    patch_ids = dense_tree.query_ball_point(center_point, r=float(patch_radius))
    if patch_ids:
        patch_ids = np.asarray(patch_ids, dtype=np.int64)
    else:
        patch_ids = np.zeros((0,), dtype=np.int64)

    if len(patch_ids) > 0 and min_normal_cos is not None:
        normal_cos = np.einsum("ij,j->i", dense_normals[patch_ids], center_normal)
        patch_ids = patch_ids[normal_cos >= float(min_normal_cos)]

    signed_depth = None
    if len(patch_ids) > 0 and patch_thickness is not None:
        offsets = dense_points[patch_ids] - center_point[None, :]
        signed_depth = np.einsum("ij,j->i", offsets, center_normal)
        keep_mask = np.abs(signed_depth) <= float(patch_thickness)
        patch_ids = patch_ids[keep_mask]
        signed_depth = signed_depth[keep_mask]

    if len(patch_ids) > 1 and patch_plane_gap_ratio is not None and patch_thickness is not None:
        plane_keep_ids = keep_patch_plane_cluster_near_center(
            signed_depth=signed_depth,
            gap_threshold=float(patch_thickness) * float(patch_plane_gap_ratio),
        )
        if len(plane_keep_ids) > 0:
            patch_ids = patch_ids[plane_keep_ids]
            signed_depth = signed_depth[plane_keep_ids]

    if len(patch_ids) > 1 and patch_dominant_normal_gap_cos is not None:
        normal_cos = np.einsum("ij,j->i", dense_normals[patch_ids], center_normal)
        dominant_keep_ids = keep_patch_dominant_normal_cluster(
            normal_cos=normal_cos,
            gap_threshold=float(patch_dominant_normal_gap_cos),
        )
        if len(dominant_keep_ids) > 0:
            patch_ids = patch_ids[dominant_keep_ids]

    if len(patch_ids) > 1 and patch_link_radius_ratio is not None:
        component_keep_ids = keep_patch_component_near_center(
            points=dense_points[patch_ids],
            center_point=center_point,
            link_radius=float(patch_radius) * float(patch_link_radius_ratio),
        )
        if len(component_keep_ids) > 0:
            patch_ids = patch_ids[component_keep_ids]

    if len(patch_ids) == 0:
        nearest_k = min(int(nearest_fallback_k), len(dense_points))
        _, nearest_ids = dense_tree.query(center_point[None, :], k=nearest_k)
        patch_ids = np.asarray(nearest_ids, dtype=np.int64).reshape(-1)
        if min_normal_cos is not None and len(patch_ids) > 0:
            normal_cos = np.einsum("ij,j->i", dense_normals[patch_ids], center_normal)
            keep_mask = normal_cos >= max(float(min_normal_cos), -0.5)
            if np.any(keep_mask):
                patch_ids = patch_ids[keep_mask]
        if len(patch_ids) > 1 and patch_dominant_normal_gap_cos is not None:
            normal_cos = np.einsum("ij,j->i", dense_normals[patch_ids], center_normal)
            dominant_keep_ids = keep_patch_dominant_normal_cluster(
                normal_cos=normal_cos,
                gap_threshold=float(patch_dominant_normal_gap_cos),
            )
            if len(dominant_keep_ids) > 0:
                patch_ids = patch_ids[dominant_keep_ids]

    return np.unique(patch_ids.astype(np.int64))


def sample_patch_points_from_ids(dense_points, patch_ids, points_per_touch, rng):
    if len(patch_ids) == 0:
        raise RuntimeError("patch_ids is empty in sample_patch_points_from_ids.")

    replace = len(patch_ids) < int(points_per_touch)
    choose_local = rng.choice(len(patch_ids), size=int(points_per_touch), replace=replace)
    return dense_points[patch_ids[choose_local]].astype(np.float32)


def sample_patch_points_and_normals_from_ids(
    dense_points,
    dense_normals,
    patch_ids,
    points_per_touch,
    rng,
):
    if len(patch_ids) == 0:
        raise RuntimeError("patch_ids is empty in sample_patch_points_and_normals_from_ids.")

    replace = len(patch_ids) < int(points_per_touch)
    choose_local = rng.choice(len(patch_ids), size=int(points_per_touch), replace=replace)
    chosen_ids = patch_ids[choose_local]
    return (
        dense_points[chosen_ids].astype(np.float32),
        dense_normals[chosen_ids].astype(np.float32),
    )


def score_candidate_coverage(
    dense_points,
    dense_normals,
    dense_tree,
    covered_mask,
    center_point,
    center_normal,
    patch_radius,
    patch_thickness,
    patch_min_normal_cos,
    patch_dominant_normal_gap_cos,
    patch_plane_gap_ratio,
    patch_link_radius_ratio,
):
    patch_ids = filter_patch_ids(
        dense_points=dense_points,
        dense_normals=dense_normals,
        dense_tree=dense_tree,
        center_point=center_point,
        center_normal=center_normal,
        patch_radius=patch_radius,
        patch_thickness=patch_thickness,
        min_normal_cos=patch_min_normal_cos,
        patch_dominant_normal_gap_cos=patch_dominant_normal_gap_cos,
        patch_plane_gap_ratio=patch_plane_gap_ratio,
        patch_link_radius_ratio=patch_link_radius_ratio,
    )
    uncovered_gain = int(np.count_nonzero(~covered_mask[patch_ids]))
    return uncovered_gain, patch_ids


def build_proposal_ids(
    candidate_points,
    candidate_tree,
    dense_points,
    covered_mask,
    used_candidate_mask,
    accepted_contact_points,
    proposal_count,
    rng,
):
    available_ids = np.flatnonzero(~used_candidate_mask)
    if len(available_ids) == 0:
        return np.zeros((0,), dtype=np.int64)

    proposal_set = set()

    random_count = min(len(available_ids), max(96, int(proposal_count) // 2))
    random_ids = rng.choice(available_ids, size=random_count, replace=False)
    proposal_set.update(np.asarray(random_ids, dtype=np.int64).tolist())

    uncovered_ids = np.flatnonzero(~covered_mask)
    if len(uncovered_ids) > 0:
        anchor_count = min(len(uncovered_ids), max(64, int(proposal_count) // 3))
        anchor_ids = rng.choice(uncovered_ids, size=anchor_count, replace=False)
        anchor_points = dense_points[anchor_ids]
        nearest_k = min(6, len(candidate_points))
        _, nearest_candidate_ids = candidate_tree.query(anchor_points, k=nearest_k)
        nearest_candidate_ids = np.asarray(nearest_candidate_ids, dtype=np.int64).reshape(-1)
        nearest_candidate_ids = nearest_candidate_ids[~used_candidate_mask[nearest_candidate_ids]]
        proposal_set.update(nearest_candidate_ids.tolist())

    if accepted_contact_points:
        accepted_points = np.asarray(accepted_contact_points, dtype=np.float32)
        probe_count = min(len(available_ids), max(96, int(proposal_count) // 3))
        probe_ids = rng.choice(available_ids, size=probe_count, replace=False)
        probe_points = candidate_points[probe_ids]
        diff = probe_points[:, None, :] - accepted_points[None, :, :]
        min_dist_sq = np.sum(diff * diff, axis=2).min(axis=1)
        far_ids = probe_ids[np.argsort(-min_dist_sq)[: min(96, len(probe_ids))]]
        proposal_set.update(np.asarray(far_ids, dtype=np.int64).tolist())

    proposal_ids = np.asarray(sorted(proposal_set), dtype=np.int64)
    if len(proposal_ids) == 0:
        return available_ids[: min(len(available_ids), int(proposal_count))]
    return proposal_ids


def candidate_payload_better(lhs, rhs):
    if rhs is None:
        return True
    if int(lhs[12]) != int(rhs[12]):
        return int(lhs[12]) > int(rhs[12])
    if float(lhs[13]) != float(rhs[13]):
        return float(lhs[13]) < float(rhs[13])
    return int(lhs[10]) > int(rhs[10])


def build_touch_view_arrays(
    accepted_patch_points,
    accepted_patch_point_normals,
    accepted_patch_centers,
    accepted_patch_center_normals,
    accepted_target_points,
    accepted_target_normals,
    accepted_contact_points,
    accepted_contact_normals,
    accepted_probe_positions,
    accepted_probe_quaternions,
    accepted_patch_source_counts,
    coverage_progress,
    num_tactile_samples,
    tactile_num_fingers,
    tactile_points_per_finger,
):
    num_views = int(num_tactile_samples)
    num_fingers = int(tactile_num_fingers)
    points_per_finger = int(tactile_points_per_finger)
    total_touches = num_views * num_fingers

    if len(accepted_patch_points) != total_touches:
        raise RuntimeError(
            f"Touch count mismatch: got {len(accepted_patch_points)}, expected {total_touches}"
        )

    view_point_count = num_fingers * points_per_finger
    touch_points = np.zeros((num_views, view_point_count, 3), dtype=np.float32)
    touch_point_normals = np.zeros((num_views, view_point_count, 3), dtype=np.float32)
    touch_round_ids = np.zeros((num_views, view_point_count), dtype=np.int32)
    touch_finger_ids = np.zeros((num_views, view_point_count), dtype=np.int32)
    touch_center_ids = np.zeros((num_views, view_point_count), dtype=np.int32)

    touch_centers = np.zeros((num_views, num_fingers, 3), dtype=np.float32)
    touch_center_normals = np.zeros((num_views, num_fingers, 3), dtype=np.float32)
    touch_target_points = np.zeros((num_views, num_fingers, 3), dtype=np.float32)
    touch_target_normals = np.zeros((num_views, num_fingers, 3), dtype=np.float32)
    touch_contact_points = np.zeros((num_views, num_fingers, 3), dtype=np.float32)
    touch_contact_normals = np.zeros((num_views, num_fingers, 3), dtype=np.float32)
    touch_probe_positions = np.zeros((num_views, num_fingers, 3), dtype=np.float32)
    touch_probe_quaternions_wxyz = np.zeros((num_views, num_fingers, 4), dtype=np.float32)
    touch_patch_source_counts = np.zeros((num_views, num_fingers), dtype=np.int32)

    for global_touch_idx in range(total_touches):
        view_idx = global_touch_idx // num_fingers
        finger_idx = global_touch_idx % num_fingers
        start = finger_idx * points_per_finger
        end = start + points_per_finger

        touch_points[view_idx, start:end] = accepted_patch_points[global_touch_idx]
        touch_point_normals[view_idx, start:end] = accepted_patch_point_normals[global_touch_idx]
        touch_finger_ids[view_idx, start:end] = finger_idx
        touch_center_ids[view_idx, start:end] = finger_idx

        touch_centers[view_idx, finger_idx] = accepted_patch_centers[global_touch_idx]
        touch_center_normals[view_idx, finger_idx] = accepted_patch_center_normals[global_touch_idx]
        touch_target_points[view_idx, finger_idx] = accepted_target_points[global_touch_idx]
        touch_target_normals[view_idx, finger_idx] = accepted_target_normals[global_touch_idx]
        touch_contact_points[view_idx, finger_idx] = accepted_contact_points[global_touch_idx]
        touch_contact_normals[view_idx, finger_idx] = accepted_contact_normals[global_touch_idx]
        touch_probe_positions[view_idx, finger_idx] = accepted_probe_positions[global_touch_idx]
        touch_probe_quaternions_wxyz[view_idx, finger_idx] = accepted_probe_quaternions[global_touch_idx]
        touch_patch_source_counts[view_idx, finger_idx] = int(accepted_patch_source_counts[global_touch_idx])

    touch_coverage_progress = np.asarray(coverage_progress, dtype=np.float32).reshape(num_views, num_fingers)
    planning_view_coverage_ratio = touch_coverage_progress[:, -1].astype(np.float32)

    return {
        "touch_points": touch_points,
        "touch_point_normals": touch_point_normals,
        "touch_round_ids": touch_round_ids,
        "touch_finger_ids": touch_finger_ids,
        "touch_center_ids": touch_center_ids,
        "touch_centers": touch_centers,
        "touch_center_normals": touch_center_normals,
        "touch_target_points": touch_target_points,
        "touch_target_normals": touch_target_normals,
        "touch_contact_points": touch_contact_points,
        "touch_contact_normals": touch_contact_normals,
        "touch_probe_positions": touch_probe_positions,
        "touch_probe_quaternions_wxyz": touch_probe_quaternions_wxyz,
        "touch_patch_source_counts": touch_patch_source_counts,
        "touch_coverage_progress": touch_coverage_progress,
        "planning_view_coverage_ratio": planning_view_coverage_ratio,
    }


def generate_mujoco_touch_data_coverage_aware(
    pipeline,
    normalized_mesh,
    normalized_mesh_path,
    num_tactile_samples=10,
    tactile_num_fingers=10,
    tactile_points_per_finger=3000,
    dense_surface_sample_n=120000,
    candidate_touch_samples=6000,
    patch_radius_ratio=0.10,
    min_touch_separation_ratio=0.055,
    patch_thickness_ratio=0.035,
    patch_min_normal_cos=0.05,
    patch_dominant_normal_gap_cos=0.18,
    patch_plane_gap_ratio=0.35,
    patch_link_radius_ratio=0.0,
    max_target_contact_offset_ratio=0.60,
    reachable_clearance_ratio=0.92,
    reachable_approach_steps=5,
    touch_mode="sphere",
    probe_geom="sphere",
    probe_radius=0.05,
    probe_capsule_half_length=0.04,
    probe_box_half_extents=None,
    approach_offset=0.18,
    indentation_depth=0.01,
    approach_steps=80,
    background_color=None,
    seed=42,
):
    if probe_box_half_extents is None:
        probe_box_half_extents = np.asarray([0.03, 0.03, 0.04], dtype=np.float32)
    else:
        probe_box_half_extents = np.asarray(probe_box_half_extents, dtype=np.float32)

    if background_color is None:
        background_color = np.asarray([0.88, 0.94, 1.0], dtype=np.float32)
    else:
        background_color = np.asarray(background_color, dtype=np.float32)

    rng = np.random.default_rng(int(seed))
    total_touches = int(num_tactile_samples) * int(tactile_num_fingers)

    raw_dense_surface_points, raw_dense_surface_normals = sample_dense_surface_points(
        normalized_mesh,
        int(dense_surface_sample_n),
    )
    diag = compute_bbox_diag(raw_dense_surface_points)

    patch_radius = float(patch_radius_ratio) * diag
    patch_thickness = float(patch_thickness_ratio) * diag
    min_touch_separation = float(min_touch_separation_ratio) * diag
    max_target_contact_offset = float(max_target_contact_offset_ratio) * patch_radius
    tactile_scene = build_raycast_scene(normalized_mesh)

    reachable_dense_mask = compute_sphere_reachability_mask(
        scene=tactile_scene,
        surface_points=raw_dense_surface_points,
        surface_normals=raw_dense_surface_normals,
        probe_radius=float(probe_radius),
        approach_offset=float(approach_offset),
        clearance_ratio=float(reachable_clearance_ratio),
        approach_steps=int(reachable_approach_steps),
    )
    reachable_dense_count = int(np.count_nonzero(reachable_dense_mask))
    minimum_dense_count = max(int(total_touches) * 32, 4096)
    if reachable_dense_count >= minimum_dense_count:
        dense_surface_points = raw_dense_surface_points[reachable_dense_mask]
        dense_surface_normals = raw_dense_surface_normals[reachable_dense_mask]
        print(
            "[INFO] tactile planning surface reachability filter kept "
            f"{reachable_dense_count}/{len(raw_dense_surface_points)} points "
            f"({reachable_dense_count / max(1, len(raw_dense_surface_points)):.4f})"
        )
    else:
        dense_surface_points = raw_dense_surface_points
        dense_surface_normals = raw_dense_surface_normals
        reachable_dense_count = int(len(raw_dense_surface_points))
        print(
            "[WARN] tactile planning reachability filter kept too few points; "
            "falling back to the unfiltered dense surface set."
        )

    dense_tree = cKDTree(dense_surface_points)

    requested_candidates = max(int(candidate_touch_samples), total_touches * 10)
    candidate_points, candidate_normals = pipeline.sample_surface_targets(
        normalized_mesh,
        num_touches=requested_candidates,
        candidate_count=requested_candidates * 2,
        seed=int(seed),
    )
    candidate_points = np.asarray(candidate_points, dtype=np.float32)
    candidate_normals = np.asarray(candidate_normals, dtype=np.float32)
    reachable_candidate_mask = compute_sphere_reachability_mask(
        scene=tactile_scene,
        surface_points=candidate_points,
        surface_normals=candidate_normals,
        probe_radius=float(probe_radius),
        approach_offset=float(approach_offset),
        clearance_ratio=float(reachable_clearance_ratio),
        approach_steps=int(reachable_approach_steps),
    )
    candidate_points = candidate_points[reachable_candidate_mask]
    candidate_normals = candidate_normals[reachable_candidate_mask]

    minimum_candidate_count = max(int(total_touches) * 8, 512)
    target_candidate_count = max(requested_candidates, minimum_candidate_count)
    if len(candidate_points) < minimum_candidate_count:
        top_up_count = target_candidate_count - len(candidate_points)
        top_up_ids = rng.choice(
            len(dense_surface_points),
            size=int(top_up_count),
            replace=len(dense_surface_points) < int(top_up_count),
        )
        candidate_points = np.concatenate(
            [candidate_points, dense_surface_points[top_up_ids].astype(np.float32)],
            axis=0,
        )
        candidate_normals = np.concatenate(
            [candidate_normals, dense_surface_normals[top_up_ids].astype(np.float32)],
            axis=0,
        )
        print(
            "[WARN] reachable candidate pool was small; topped up candidates from the "
            "reachable tactile planning surface."
        )

    candidate_tree = cKDTree(candidate_points)
    used_candidate_mask = np.zeros(len(candidate_points), dtype=bool)

    if touch_mode != "sphere":
        raise ValueError(
            "This MuJoCo batch script currently supports touch_mode='sphere' only. "
            "If you want ur5_ee / ur5_arm, I can open another version."
        )

    model, data, probe_joint_id, probe_geom_id, object_geom_id = pipeline.build_mujoco_model(
        Path(normalized_mesh_path),
        probe_geom=probe_geom,
        probe_radius=float(probe_radius),
        probe_capsule_half_length=float(probe_capsule_half_length),
        probe_box_half_extents=probe_box_half_extents,
        background_color=background_color,
    )

    covered_mask = np.zeros(len(dense_surface_points), dtype=bool)
    accepted_patch_points = []
    accepted_patch_point_normals = []
    accepted_patch_centers = []
    accepted_patch_center_normals = []
    accepted_target_points = []
    accepted_target_normals = []
    accepted_contact_points = []
    accepted_contact_normals = []
    accepted_probe_positions = []
    accepted_probe_quaternions = []
    accepted_patch_source_counts = []
    coverage_progress = []

    for touch_slot in range(total_touches):
        proposal_ids = build_proposal_ids(
            candidate_points=candidate_points,
            candidate_tree=candidate_tree,
            dense_points=dense_surface_points,
            covered_mask=covered_mask,
            used_candidate_mask=used_candidate_mask,
            accepted_contact_points=accepted_contact_points,
            proposal_count=320,
            rng=rng,
        )

        if len(proposal_ids) == 0:
            if len(candidate_points) > 0 and np.any(used_candidate_mask):
                print(
                    f"[WARN] candidate proposal pool exhausted at slot={touch_slot:03d}; "
                    "recycling proposal ids for repeat-touch fallback."
                )
                used_candidate_mask[:] = False
                proposal_ids = build_proposal_ids(
                    candidate_points=candidate_points,
                    candidate_tree=candidate_tree,
                    dense_points=dense_surface_points,
                    covered_mask=covered_mask,
                    used_candidate_mask=used_candidate_mask,
                    accepted_contact_points=accepted_contact_points,
                    proposal_count=320,
                    rng=rng,
                )

        if len(proposal_ids) == 0:
            raise RuntimeError(
                f"No candidate proposals remain before touch slot {touch_slot}."
            )

        scored_candidates = []
        for proposal_idx in proposal_ids:
            if used_candidate_mask[proposal_idx]:
                continue
            uncovered_gain, _ = score_candidate_coverage(
                dense_points=dense_surface_points,
                dense_normals=dense_surface_normals,
                dense_tree=dense_tree,
                covered_mask=covered_mask,
                center_point=candidate_points[proposal_idx],
                center_normal=normalize_vector(candidate_normals[proposal_idx]),
                patch_radius=patch_radius,
                patch_thickness=patch_thickness,
                patch_min_normal_cos=patch_min_normal_cos,
                patch_dominant_normal_gap_cos=patch_dominant_normal_gap_cos,
                patch_plane_gap_ratio=patch_plane_gap_ratio,
                patch_link_radius_ratio=patch_link_radius_ratio,
            )
            scored_candidates.append((uncovered_gain, int(proposal_idx)))

        scored_candidates.sort(reverse=True)
        attempt_order = [idx for _, idx in scored_candidates[:96]]

        if len(attempt_order) == 0:
            available_ids = np.flatnonzero(~used_candidate_mask)
            attempt_order = available_ids[: min(len(available_ids), 96)].tolist()

        accepted_this_touch = False
        best_repeat_candidate = None
        best_relaxed_candidate = None
        best_separation_relaxed_candidate = None

        for proposal_idx in attempt_order:
            if used_candidate_mask[proposal_idx]:
                continue
            used_candidate_mask[proposal_idx] = True

            target_point = candidate_points[proposal_idx]
            target_normal = normalize_vector(candidate_normals[proposal_idx])

            contact_result = pipeline.simulate_touch_contact(
                model=model,
                data=data,
                probe_joint_id=probe_joint_id,
                probe_geom_id=probe_geom_id,
                object_geom_id=object_geom_id,
                target_point=target_point,
                outward_normal=target_normal,
                touch_mode=touch_mode,
                probe_geom=probe_geom,
                probe_radius=float(probe_radius),
                probe_capsule_half_length=float(probe_capsule_half_length),
                probe_box_half_extents=probe_box_half_extents,
                approach_offset=float(approach_offset),
                indentation_depth=float(indentation_depth),
                approach_steps=int(approach_steps),
                ur5_roll_jitter_deg=0.0,
                rng=rng,
                viewer=None,
                viewer_sleep=0.0,
            )
            if contact_result is None:
                continue

            contact_point, contact_normal, probe_position, probe_quaternion = contact_result
            contact_point = np.asarray(contact_point, dtype=np.float32)
            contact_normal = align_normal_to_reference(contact_normal, target_normal)
            probe_position = np.asarray(probe_position, dtype=np.float32)
            probe_quaternion = np.asarray(probe_quaternion, dtype=np.float32)

            target_contact_offset = float(
                np.linalg.norm(contact_point.astype(np.float32) - target_point.astype(np.float32))
            )
            patch_center_point = target_point.astype(np.float32)
            patch_center_normal = target_normal.astype(np.float32)
            patch_ids = filter_patch_ids(
                dense_points=dense_surface_points,
                dense_normals=dense_surface_normals,
                dense_tree=dense_tree,
                center_point=patch_center_point,
                center_normal=patch_center_normal,
                patch_radius=float(patch_radius),
                patch_thickness=float(patch_thickness),
                min_normal_cos=patch_min_normal_cos,
                patch_dominant_normal_gap_cos=patch_dominant_normal_gap_cos,
                patch_plane_gap_ratio=patch_plane_gap_ratio,
                patch_link_radius_ratio=patch_link_radius_ratio,
            )
            patch_points, patch_point_normals = sample_patch_points_and_normals_from_ids(
                dense_points=dense_surface_points,
                dense_normals=dense_surface_normals,
                patch_ids=patch_ids,
                points_per_touch=int(tactile_points_per_finger),
                rng=rng,
            )
            source_count = int(len(patch_ids))
            uncovered_gain = int(np.count_nonzero(~covered_mask[patch_ids]))

            candidate_payload = (
                patch_points.astype(np.float32),
                patch_point_normals.astype(np.float32),
                patch_center_point.astype(np.float32),
                patch_center_normal.astype(np.float32),
                target_point.astype(np.float32),
                target_normal.astype(np.float32),
                contact_point.astype(np.float32),
                contact_normal.astype(np.float32),
                probe_position.astype(np.float32),
                probe_quaternion.astype(np.float32),
                int(source_count),
                np.asarray(patch_ids, dtype=np.int64),
                uncovered_gain,
                float(target_contact_offset),
            )

            if accepted_contact_points:
                min_sep = np.min(
                    np.linalg.norm(
                        np.asarray(accepted_contact_points, dtype=np.float32) - contact_point[None, :],
                        axis=1,
                    )
                )
                if min_sep < float(min_touch_separation):
                    if candidate_payload_better(candidate_payload, best_separation_relaxed_candidate):
                        best_separation_relaxed_candidate = candidate_payload
                    continue

            if target_contact_offset > float(max_target_contact_offset):
                if candidate_payload_better(candidate_payload, best_relaxed_candidate):
                    best_relaxed_candidate = candidate_payload
                continue

            if uncovered_gain > 0:
                (
                    patch_points,
                    patch_point_normals,
                    patch_center_point,
                    patch_center_normal,
                    target_point,
                    target_normal,
                    contact_point,
                    contact_normal,
                    probe_position,
                    probe_quaternion,
                    source_count,
                    patch_ids,
                    uncovered_gain,
                    target_contact_offset,
                ) = candidate_payload
                accepted_this_touch = True
                break

            if candidate_payload_better(candidate_payload, best_repeat_candidate):
                best_repeat_candidate = candidate_payload

        if not accepted_this_touch:
            fallback_reason = None
            if best_repeat_candidate is not None:
                fallback_reason = "repeat"
                fallback_candidate = best_repeat_candidate
            elif best_relaxed_candidate is not None:
                fallback_reason = "relaxed-offset"
                fallback_candidate = best_relaxed_candidate
            elif best_separation_relaxed_candidate is not None:
                fallback_reason = "relaxed-separation"
                fallback_candidate = best_separation_relaxed_candidate
            else:
                raise RuntimeError(
                    f"Failed to simulate a valid MuJoCo touch for slot {touch_slot}."
                )
            (
                patch_points,
                patch_point_normals,
                patch_center_point,
                patch_center_normal,
                target_point,
                target_normal,
                contact_point,
                contact_normal,
                probe_position,
                probe_quaternion,
                source_count,
                patch_ids,
                uncovered_gain,
                target_contact_offset,
            ) = fallback_candidate
            print(
                f"[WARN] using {fallback_reason} fallback at slot={touch_slot:03d} "
                f"offset={float(target_contact_offset):.6f}"
            )

        accepted_patch_points.append(patch_points)
        accepted_patch_point_normals.append(patch_point_normals)
        accepted_patch_centers.append(patch_center_point)
        accepted_patch_center_normals.append(patch_center_normal)
        accepted_target_points.append(target_point)
        accepted_target_normals.append(target_normal)
        accepted_contact_points.append(contact_point)
        accepted_contact_normals.append(contact_normal)
        accepted_probe_positions.append(probe_position)
        accepted_probe_quaternions.append(probe_quaternion)
        accepted_patch_source_counts.append(int(source_count))

        covered_mask[patch_ids] = True
        current_coverage = float(np.mean(covered_mask))
        coverage_progress.append(current_coverage)

        print(
            f"[TOUCH mujoco-coverage] slot={touch_slot:03d} "
            f"new_cover={uncovered_gain:06d} coverage={current_coverage:.4f} "
            f"source_points={int(source_count)} "
            f"target_contact_offset={float(target_contact_offset):.6f}"
        )

    touch_data = build_touch_view_arrays(
        accepted_patch_points=accepted_patch_points,
        accepted_patch_point_normals=accepted_patch_point_normals,
        accepted_patch_centers=accepted_patch_centers,
        accepted_patch_center_normals=accepted_patch_center_normals,
        accepted_target_points=accepted_target_points,
        accepted_target_normals=accepted_target_normals,
        accepted_contact_points=accepted_contact_points,
        accepted_contact_normals=accepted_contact_normals,
        accepted_probe_positions=accepted_probe_positions,
        accepted_probe_quaternions=accepted_probe_quaternions,
        accepted_patch_source_counts=accepted_patch_source_counts,
        coverage_progress=coverage_progress,
        num_tactile_samples=num_tactile_samples,
        tactile_num_fingers=tactile_num_fingers,
        tactile_points_per_finger=tactile_points_per_finger,
    )
    touch_data["planning_surface_coverage_ratio"] = np.array(
        coverage_progress[-1] if coverage_progress else 0.0,
        dtype=np.float32,
    )
    touch_data["planning_dense_surface_point_count"] = np.array(
        len(raw_dense_surface_points),
        dtype=np.int32,
    )
    touch_data["planning_reachable_surface_point_count"] = np.array(
        len(dense_surface_points),
        dtype=np.int32,
    )
    touch_data["planning_reachable_surface_fraction"] = np.array(
        float(len(dense_surface_points)) / float(max(1, len(raw_dense_surface_points))),
        dtype=np.float32,
    )
    touch_data["planning_candidate_point_count"] = np.array(
        len(candidate_points),
        dtype=np.int32,
    )
    return touch_data


def process_single_obj_to_mujoco_coverage_npz(
    obj_path,
    out_path,
    normalized_mesh_asset_path,
    num_tactile_samples=10,
    tactile_num_fingers=10,
    tactile_points_per_finger=3000,
    dense_surface_sample_n=120000,
    candidate_touch_samples=6000,
    tactile_patch_radius_ratio=0.10,
    tactile_min_touch_separation_ratio=0.055,
    tactile_patch_thickness_ratio=0.035,
    patch_min_normal_cos=0.05,
    tactile_patch_dominant_normal_gap_cos=0.18,
    tactile_patch_plane_gap_ratio=0.35,
    tactile_patch_link_radius_ratio=0.0,
    max_target_contact_offset_ratio=0.60,
    tactile_reachable_clearance_ratio=0.92,
    tactile_reachable_approach_steps=5,
    normalization_bound=0.9,
    num_surface_points=235000,
    num_query_points=250000,
    query_uniform_region="cube",
    seed=42,
):
    print("\n==================================================")
    print("[PROCESS mujoco coverage-aware]", obj_path)
    print("==================================================")

    pipeline = load_tactistruct_pipeline_module()
    mesh_name = os.path.splitext(os.path.basename(obj_path))[0]

    source_mesh = pipeline.load_input_mesh(Path(obj_path))
    normalized_mesh, transform = pipeline.normalize_mesh(
        source_mesh,
        float(normalization_bound),
    )

    normalized_mesh_asset_path = Path(normalized_mesh_asset_path)
    normalized_mesh_asset_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_mesh.export(normalized_mesh_asset_path)

    surface_points, surface_normals = sample_surface_points_for_storage(
        normalized_mesh,
        num_surface_points=num_surface_points,
    )

    scene = build_raycast_scene(normalized_mesh)

    print(f"[INFO] sampling query points near the surface (uniform_region={query_uniform_region}) ...")
    query_points = sample_query_points_near_surface(
        surface_points=surface_points,
        number_of_points=num_query_points,
        uniform_region=query_uniform_region,
    )

    if normalized_mesh.is_watertight:
        print("[INFO] computing query_sdf using RaycastingScene occupancy sign (watertight mesh) ...")
    else:
        print("[INFO] computing query_sdf using unsigned distance + near-surface sign fallback (non-watertight mesh) ...")

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
        normalized_mesh_path=normalized_mesh_asset_path,
        num_tactile_samples=num_tactile_samples,
        tactile_num_fingers=tactile_num_fingers,
        tactile_points_per_finger=tactile_points_per_finger,
        dense_surface_sample_n=dense_surface_sample_n,
        candidate_touch_samples=candidate_touch_samples,
        patch_radius_ratio=tactile_patch_radius_ratio,
        min_touch_separation_ratio=tactile_min_touch_separation_ratio,
        patch_thickness_ratio=tactile_patch_thickness_ratio,
        patch_min_normal_cos=patch_min_normal_cos,
        patch_dominant_normal_gap_cos=tactile_patch_dominant_normal_gap_cos,
        patch_plane_gap_ratio=tactile_patch_plane_gap_ratio,
        patch_link_radius_ratio=tactile_patch_link_radius_ratio,
        max_target_contact_offset_ratio=max_target_contact_offset_ratio,
        reachable_clearance_ratio=tactile_reachable_clearance_ratio,
        reachable_approach_steps=tactile_reachable_approach_steps,
        touch_mode="sphere",
        probe_geom="sphere",
        probe_radius=0.05,
        probe_capsule_half_length=0.04,
        probe_box_half_extents=np.asarray([0.03, 0.03, 0.04], dtype=np.float32),
        approach_offset=0.18,
        indentation_depth=0.01,
        approach_steps=80,
        background_color=np.asarray([0.88, 0.94, 1.0], dtype=np.float32),
        seed=int(seed),
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    np.savez_compressed(
        out_path,
        surface_points=surface_points,
        surface_normals=surface_normals,
        query_points=query_points.astype(np.float32),
        query_sdf=query_sdf.astype(np.float32),
        touch_points=touch_data["touch_points"],
        touch_point_normals=touch_data["touch_point_normals"],
        touch_round_ids=touch_data["touch_round_ids"],
        touch_finger_ids=touch_data["touch_finger_ids"],
        touch_center_ids=touch_data["touch_center_ids"],
        touch_centers=touch_data["touch_centers"],
        touch_center_normals=touch_data["touch_center_normals"],
        touch_target_points=touch_data["touch_target_points"],
        touch_target_normals=touch_data["touch_target_normals"],
        touch_contact_points=touch_data["touch_contact_points"],
        touch_contact_normals=touch_data["touch_contact_normals"],
        touch_probe_positions=touch_data["touch_probe_positions"],
        touch_probe_quaternions_wxyz=touch_data["touch_probe_quaternions_wxyz"],
        touch_patch_source_counts=touch_data["touch_patch_source_counts"],
        touch_coverage_progress=touch_data["touch_coverage_progress"],
        planning_surface_coverage_ratio=touch_data["planning_surface_coverage_ratio"],
        planning_view_coverage_ratio=touch_data["planning_view_coverage_ratio"],
        planning_dense_surface_point_count=touch_data["planning_dense_surface_point_count"],
        planning_reachable_surface_point_count=touch_data["planning_reachable_surface_point_count"],
        planning_reachable_surface_fraction=touch_data["planning_reachable_surface_fraction"],
        planning_candidate_point_count=touch_data["planning_candidate_point_count"],
        object_center=transform.center.astype(np.float32),
        object_scale=np.asarray(transform.scale, dtype=np.float32),
        normalization_bound=np.asarray(transform.target_bound, dtype=np.float32),
        source_mesh=np.asarray(str(obj_path)),
        normalized_mesh_asset=np.asarray(str(normalized_mesh_asset_path)),
        mesh_name=np.array(mesh_name),
        num_tactile_samples=np.array(num_tactile_samples, dtype=np.int32),
        tactile_num_fingers=np.array(tactile_num_fingers, dtype=np.int32),
        query_uniform_region=np.asarray(str(query_uniform_region)),
    )

    print("[SAVED]", out_path)
    print("surface_points                 :", surface_points.shape)
    print("query_points                   :", query_points.shape)
    print("query_sdf                      :", query_sdf.shape)
    print("touch_points                   :", touch_data["touch_points"].shape)
    print("touch_point_normals            :", touch_data["touch_point_normals"].shape)
    print("touch_centers                  :", touch_data["touch_centers"].shape)
    print(
        f"planning_surface_coverage_ratio: "
        f"{float(touch_data['planning_surface_coverage_ratio']):.4f}"
    )
    print(
        "planning_reachable_surface_fraction: "
        f"{float(touch_data['planning_reachable_surface_fraction']):.4f}"
    )

    return {
        "out_path": str(out_path),
        "normalized_mesh_asset_path": str(normalized_mesh_asset_path),
        "planning_surface_coverage_ratio": float(
            touch_data["planning_surface_coverage_ratio"]
        ),
        "planning_reachable_surface_fraction": float(
            touch_data["planning_reachable_surface_fraction"]
        ),
        "num_touch_views": int(touch_data["touch_points"].shape[0]),
    }


def process_single_obj_job(job):
    obj_path = str(job["obj_path"])
    out_path = str(job["out_path"])
    normalized_mesh_asset_path = str(job["normalized_mesh_asset_path"])

    if os.path.exists(out_path) and not bool(job["overwrite"]):
        return {
            "status": "skipped",
            "obj_path": obj_path,
            "out_path": out_path,
            "message": "exists",
        }

    try:
        result = process_single_obj_to_mujoco_coverage_npz(
            obj_path=obj_path,
            out_path=out_path,
            normalized_mesh_asset_path=normalized_mesh_asset_path,
            num_tactile_samples=int(job["num_tactile_samples"]),
            tactile_num_fingers=int(job["tactile_num_fingers"]),
            tactile_points_per_finger=int(job["tactile_points_per_finger"]),
            dense_surface_sample_n=int(job["dense_surface_sample_n"]),
            candidate_touch_samples=int(job["candidate_touch_samples"]),
            tactile_patch_radius_ratio=float(job["tactile_patch_radius_ratio"]),
            tactile_min_touch_separation_ratio=float(
                job["tactile_min_touch_separation_ratio"]
            ),
            tactile_patch_thickness_ratio=float(job["tactile_patch_thickness_ratio"]),
            patch_min_normal_cos=float(job["patch_min_normal_cos"]),
            tactile_patch_dominant_normal_gap_cos=float(job["tactile_patch_dominant_normal_gap_cos"]),
            tactile_patch_plane_gap_ratio=float(job["tactile_patch_plane_gap_ratio"]),
            tactile_patch_link_radius_ratio=float(job["tactile_patch_link_radius_ratio"]),
            max_target_contact_offset_ratio=float(job["max_target_contact_offset_ratio"]),
            tactile_reachable_clearance_ratio=float(job["tactile_reachable_clearance_ratio"]),
            tactile_reachable_approach_steps=int(job["tactile_reachable_approach_steps"]),
            normalization_bound=float(job["normalization_bound"]),
            num_surface_points=int(job["num_surface_points"]),
            num_query_points=int(job["num_query_points"]),
            query_uniform_region=str(job["query_uniform_region"]),
            seed=int(job["seed"]),
        )
        return {
            "status": "ok",
            "obj_path": obj_path,
            "out_path": out_path,
            "coverage": float(result["planning_surface_coverage_ratio"]),
            "reachable_fraction": float(result["planning_reachable_surface_fraction"]),
            "num_touch_views": int(result["num_touch_views"]),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "obj_path": obj_path,
            "out_path": out_path,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def build_jobs(
    root_dir,
    category_names=None,
    max_objects_per_category=None,
    num_tactile_samples=10,
    tactile_num_fingers=10,
    tactile_points_per_finger=3000,
    dense_surface_sample_n=120000,
    candidate_touch_samples=6000,
    tactile_patch_radius_ratio=0.10,
    tactile_min_touch_separation_ratio=0.055,
    tactile_patch_thickness_ratio=0.035,
    patch_min_normal_cos=0.05,
    tactile_patch_dominant_normal_gap_cos=0.18,
    tactile_patch_plane_gap_ratio=0.35,
    tactile_patch_link_radius_ratio=0.0,
    max_target_contact_offset_ratio=0.60,
    tactile_reachable_clearance_ratio=0.92,
    tactile_reachable_approach_steps=5,
    normalization_bound=0.9,
    num_surface_points=235000,
    num_query_points=250000,
    query_uniform_region="cube",
    output_folder_name="tactistruct_npz_shapenet_mujoco_coverage_onefolder",
    asset_folder_name="tactistruct_npz_shapenet_mujoco_coverage_assets",
    overwrite=False,
    base_seed=42,
):
    category_dirs = list(iter_category_dirs(root_dir, category_names=category_names))
    jobs = []

    for category_dir in category_dirs:
        obj_paths = find_shapenet_obj_files(
            category_dir,
            max_objects=max_objects_per_category,
        )
        for obj_path in obj_paths:
            out_path = build_flat_output_path(
                obj_path=obj_path,
                root_dir=root_dir,
                output_folder_name=output_folder_name,
            )
            normalized_mesh_asset_path = build_asset_export_path(
                obj_path=obj_path,
                root_dir=root_dir,
                asset_folder_name=asset_folder_name,
            )
            jobs.append(
                {
                    "obj_path": obj_path,
                    "out_path": out_path,
                    "normalized_mesh_asset_path": normalized_mesh_asset_path,
                    "overwrite": bool(overwrite),
                    "num_tactile_samples": int(num_tactile_samples),
                    "tactile_num_fingers": int(tactile_num_fingers),
                    "tactile_points_per_finger": int(tactile_points_per_finger),
                    "dense_surface_sample_n": int(dense_surface_sample_n),
                    "candidate_touch_samples": int(candidate_touch_samples),
                    "tactile_patch_radius_ratio": float(tactile_patch_radius_ratio),
                    "tactile_min_touch_separation_ratio": float(
                        tactile_min_touch_separation_ratio
                    ),
                    "tactile_patch_thickness_ratio": float(tactile_patch_thickness_ratio),
                    "patch_min_normal_cos": float(patch_min_normal_cos),
                    "tactile_patch_dominant_normal_gap_cos": float(tactile_patch_dominant_normal_gap_cos),
                    "tactile_patch_plane_gap_ratio": float(tactile_patch_plane_gap_ratio),
                    "tactile_patch_link_radius_ratio": float(tactile_patch_link_radius_ratio),
                    "max_target_contact_offset_ratio": float(max_target_contact_offset_ratio),
                    "tactile_reachable_clearance_ratio": float(tactile_reachable_clearance_ratio),
                    "tactile_reachable_approach_steps": int(tactile_reachable_approach_steps),
                    "normalization_bound": float(normalization_bound),
                    "num_surface_points": int(num_surface_points),
                    "num_query_points": int(num_query_points),
                    "query_uniform_region": str(query_uniform_region),
                    "seed": int(object_seed(base_seed, obj_path)),
                }
            )
    return category_dirs, jobs


def run_parallel_jobs(jobs, max_workers=1, fail_fast=False):
    if not jobs:
        return []

    max_workers = max(1, int(max_workers))
    if max_workers == 1:
        return [process_single_obj_job(job) for job in jobs]

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(process_single_obj_job, job): job for job in jobs
        }
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
            if result["status"] == "failed" and bool(fail_fast):
                for pending in future_to_job:
                    pending.cancel()
                break
    return results


def summarise_results(results):
    summary = {
        "ok": 0,
        "skipped": 0,
        "failed": 0,
        "mean_coverage": 0.0,
        "mean_reachable_surface_fraction": 0.0,
        "outputs": [],
        "failures": [],
    }
    coverages = []
    reachable_fractions = []
    for result in results:
        status = result["status"]
        summary[status] += 1
        if status == "ok":
            summary["outputs"].append(result["out_path"])
            coverages.append(float(result.get("coverage", 0.0)))
            reachable_fractions.append(float(result.get("reachable_fraction", 0.0)))
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
    if reachable_fractions:
        summary["mean_reachable_surface_fraction"] = float(np.mean(reachable_fractions))
    return summary


def process_shapenetcore_all_categories_mujoco_coverage_onefolder(
    root_dir,
    category_names=None,
    max_objects_per_category=None,
    num_tactile_samples=10,
    tactile_num_fingers=10,
    tactile_points_per_finger=3000,
    dense_surface_sample_n=120000,
    candidate_touch_samples=6000,
    tactile_patch_radius_ratio=0.10,
    tactile_min_touch_separation_ratio=0.055,
    tactile_patch_thickness_ratio=0.035,
    patch_min_normal_cos=0.05,
    tactile_patch_dominant_normal_gap_cos=0.18,
    tactile_patch_plane_gap_ratio=0.35,
    tactile_patch_link_radius_ratio=0.0,
    max_target_contact_offset_ratio=0.60,
    tactile_reachable_clearance_ratio=0.92,
    tactile_reachable_approach_steps=5,
    normalization_bound=0.9,
    num_surface_points=235000,
    num_query_points=250000,
    query_uniform_region="cube",
    output_folder_name="tactistruct_npz_shapenet_mujoco_coverage_onefolder",
    asset_folder_name="tactistruct_npz_shapenet_mujoco_coverage_assets",
    max_workers=1,
    overwrite=False,
    fail_fast=False,
    base_seed=42,
):
    category_dirs, jobs = build_jobs(
        root_dir=root_dir,
        category_names=category_names,
        max_objects_per_category=max_objects_per_category,
        num_tactile_samples=num_tactile_samples,
        tactile_num_fingers=tactile_num_fingers,
        tactile_points_per_finger=tactile_points_per_finger,
        dense_surface_sample_n=dense_surface_sample_n,
        candidate_touch_samples=candidate_touch_samples,
        tactile_patch_radius_ratio=tactile_patch_radius_ratio,
        tactile_min_touch_separation_ratio=tactile_min_touch_separation_ratio,
        tactile_patch_thickness_ratio=tactile_patch_thickness_ratio,
        patch_min_normal_cos=patch_min_normal_cos,
        tactile_patch_dominant_normal_gap_cos=tactile_patch_dominant_normal_gap_cos,
        tactile_patch_plane_gap_ratio=tactile_patch_plane_gap_ratio,
        tactile_patch_link_radius_ratio=tactile_patch_link_radius_ratio,
        max_target_contact_offset_ratio=max_target_contact_offset_ratio,
        tactile_reachable_clearance_ratio=tactile_reachable_clearance_ratio,
        tactile_reachable_approach_steps=tactile_reachable_approach_steps,
        normalization_bound=normalization_bound,
        num_surface_points=num_surface_points,
        num_query_points=num_query_points,
        query_uniform_region=query_uniform_region,
        output_folder_name=output_folder_name,
        asset_folder_name=asset_folder_name,
        overwrite=overwrite,
        base_seed=base_seed,
    )

    if not category_dirs:
        print("[WARN] no category folders found under:", root_dir)
        return {"ok": 0, "skipped": 0, "failed": 0, "mean_coverage": 0.0}

    out_dir = os.path.join(root_dir, output_folder_name)
    asset_dir = os.path.join(root_dir, asset_folder_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(asset_dir, exist_ok=True)

    print(f"[INFO] found {len(category_dirs)} category folders under root.")
    print(f"[INFO] flat output dir : {out_dir}")
    print(f"[INFO] MuJoCo asset dir: {asset_dir}")
    print(f"[INFO] prepared {len(jobs)} object jobs")
    print(f"[INFO] max_workers={int(max_workers)} overwrite={bool(overwrite)}")

    results = run_parallel_jobs(
        jobs=jobs,
        max_workers=max_workers,
        fail_fast=fail_fast,
    )
    for result in results:
        status = result["status"].upper()
        if result["status"] == "ok":
            print(
                f"[{status}] {result['obj_path']} -> {result['out_path']} "
                f"(coverage={result['coverage']:.4f}, "
                f"reachable_fraction={result['reachable_fraction']:.4f})"
            )
        elif result["status"] == "skipped":
            print(f"[{status}] {result['obj_path']} ({result['message']})")
        else:
            print(f"[{status}] {result['obj_path']}")
            print(result.get("error", "unknown error"))

    summary = summarise_results(results)
    summary_path = os.path.join(out_dir, "preprocess_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        import json

        json.dump(summary, handle, indent=2)
    print(f"[SUMMARY] saved to {summary_path}")
    print(
        f"[SUMMARY] ok={summary['ok']} skipped={summary['skipped']} "
        f"failed={summary['failed']} mean_coverage={summary['mean_coverage']:.4f} "
        f"mean_reachable_surface_fraction={summary['mean_reachable_surface_fraction']:.4f}"
    )
    if summary["failed"] > 0 and bool(fail_fast):
        raise RuntimeError("At least one preprocessing job failed in fail-fast mode.")
    return summary


if __name__ == "__main__":
    multiprocessing.freeze_support()
    args = parse_args()
    process_shapenetcore_all_categories_mujoco_coverage_onefolder(
        root_dir=str(Path(args.root_dir).resolve()),
        category_names=parse_category_names(args.category_names),
        max_objects_per_category=args.max_objects_per_category,
        num_tactile_samples=args.num_tactile_samples,
        tactile_num_fingers=args.tactile_num_fingers,
        tactile_points_per_finger=args.tactile_points_per_finger,
        dense_surface_sample_n=args.dense_surface_sample_n,
        candidate_touch_samples=args.candidate_touch_samples,
        tactile_patch_radius_ratio=args.tactile_patch_radius_ratio,
        tactile_min_touch_separation_ratio=args.tactile_min_touch_separation_ratio,
        tactile_patch_thickness_ratio=args.tactile_patch_thickness_ratio,
        patch_min_normal_cos=args.patch_min_normal_cos,
        tactile_patch_dominant_normal_gap_cos=args.tactile_patch_dominant_normal_gap_cos,
        tactile_patch_plane_gap_ratio=args.tactile_patch_plane_gap_ratio,
        tactile_patch_link_radius_ratio=args.tactile_patch_link_radius_ratio,
        max_target_contact_offset_ratio=args.max_target_contact_offset_ratio,
        tactile_reachable_clearance_ratio=args.tactile_reachable_clearance_ratio,
        tactile_reachable_approach_steps=args.tactile_reachable_approach_steps,
        normalization_bound=args.normalization_bound,
        num_surface_points=args.num_surface_points,
        num_query_points=args.num_query_points,
        query_uniform_region=str(args.query_uniform_region),
        output_folder_name=args.output_folder_name,
        asset_folder_name=args.asset_folder_name,
        max_workers=args.max_workers,
        overwrite=bool(args.overwrite),
        fail_fast=bool(args.fail_fast),
        base_seed=args.base_seed,
    )
    print("\nAll done.")
