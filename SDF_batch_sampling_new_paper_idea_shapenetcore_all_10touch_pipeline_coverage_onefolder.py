import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from SDF_batch_sampling_new_paper_idea import (
    build_raycast_scene,
    compute_query_sdf_with_raycasting,
    sample_query_points_near_surface,
    sample_surface_points_for_storage,
    scale_to_unit_sphere,
)
from SDF_batch_sampling_new_paper_idea_shapenetcore_all import (
    find_shapenet_obj_files,
    iter_category_dirs,
)


def build_flat_output_path(obj_path, root_dir, output_folder_name):
    rel_path = os.path.relpath(obj_path, root_dir)
    rel_without_ext = os.path.splitext(rel_path)[0]
    safe_name = rel_without_ext.replace("\\", "__").replace("/", "__").replace(":", "_")
    out_dir = os.path.join(root_dir, output_folder_name)
    return os.path.join(out_dir, safe_name + ".npz")


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


def sample_candidate_touch_targets(mesh, candidate_count, seed):
    rng = np.random.default_rng(seed)
    points, face_ids = trimesh.sample.sample_surface(mesh, int(candidate_count))
    points = points.astype(np.float32)
    normals = mesh.face_normals[face_ids].astype(np.float32)

    centroid = mesh.bounding_box.centroid.astype(np.float32)
    outward_hint = points - centroid[None, :]
    flip_mask = np.einsum("ij,ij->i", normals, outward_hint) < 0.0
    normals[flip_mask] *= -1.0

    order = rng.permutation(len(points))
    return points[order], normals[order]


def filter_patch_ids(
    dense_points,
    dense_normals,
    dense_tree,
    center_point,
    center_normal,
    patch_radius,
    patch_thickness,
    min_normal_cos=0.0,
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

    if len(patch_ids) > 0 and patch_thickness is not None:
        offsets = dense_points[patch_ids] - center_point[None, :]
        signed_depth = np.abs(np.einsum("ij,j->i", offsets, center_normal))
        patch_ids = patch_ids[signed_depth <= float(patch_thickness)]

    if len(patch_ids) == 0:
        nearest_k = min(int(nearest_fallback_k), len(dense_points))
        _, nearest_ids = dense_tree.query(center_point[None, :], k=nearest_k)
        patch_ids = np.asarray(nearest_ids, dtype=np.int64).reshape(-1)
        if min_normal_cos is not None and len(patch_ids) > 0:
            normal_cos = np.einsum("ij,j->i", dense_normals[patch_ids], center_normal)
            keep_mask = normal_cos >= max(float(min_normal_cos), -0.5)
            if np.any(keep_mask):
                patch_ids = patch_ids[keep_mask]

    return np.unique(patch_ids.astype(np.int64))


def sample_patch_points_from_ids(dense_points, patch_ids, points_per_touch, rng):
    if len(patch_ids) == 0:
        raise RuntimeError("patch_ids is empty in sample_patch_points_from_ids.")

    replace = len(patch_ids) < int(points_per_touch)
    choose_local = rng.choice(len(patch_ids), size=int(points_per_touch), replace=replace)
    return dense_points[patch_ids[choose_local]].astype(np.float32)


def pick_rescue_center(
    dense_points,
    dense_normals,
    dense_tree,
    uncovered_mask,
    selected_points,
    patch_radius,
    patch_thickness,
    min_normal_cos,
    rng,
):
    uncovered_ids = np.flatnonzero(uncovered_mask)
    if len(uncovered_ids) == 0:
        return None

    sample_size = min(len(uncovered_ids), 2048)
    probe_ids = rng.choice(uncovered_ids, size=sample_size, replace=False)
    probe_points = dense_points[probe_ids]

    if len(selected_points) == 0:
        best_idx = int(probe_ids[rng.integers(len(probe_ids))])
    else:
        selected_points = np.asarray(selected_points, dtype=np.float32)
        diff = probe_points[:, None, :] - selected_points[None, :, :]
        min_dist_sq = np.sum(diff * diff, axis=2).min(axis=1)
        best_idx = int(probe_ids[np.argmax(min_dist_sq)])

    center_point = dense_points[best_idx]
    center_normal = dense_normals[best_idx]
    patch_ids = filter_patch_ids(
        dense_points=dense_points,
        dense_normals=dense_normals,
        dense_tree=dense_tree,
        center_point=center_point,
        center_normal=center_normal,
        patch_radius=patch_radius,
        patch_thickness=patch_thickness,
        min_normal_cos=min_normal_cos,
    )
    if len(patch_ids) == 0:
        return None

    return center_point, center_normal, patch_ids


def greedy_select_touch_targets(
    dense_points,
    dense_normals,
    candidate_points,
    candidate_normals,
    total_centers,
    patch_radius,
    patch_thickness,
    min_center_separation,
    min_normal_cos=0.0,
    proposal_count=256,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    dense_tree = cKDTree(dense_points)
    candidate_tree = cKDTree(candidate_points)
    covered_mask = np.zeros(len(dense_points), dtype=bool)
    selected_centers = []
    selected_normals = []
    selected_patch_ids = []
    coverage_progress = []

    num_candidates = len(candidate_points)
    if num_candidates == 0:
        raise RuntimeError("candidate_points is empty in greedy_select_touch_targets.")

    for center_slot in range(int(total_centers)):
        proposal_set = set()

        random_count = min(num_candidates, max(64, int(proposal_count) // 2))
        random_ids = rng.choice(num_candidates, size=random_count, replace=False)
        proposal_set.update(np.asarray(random_ids, dtype=np.int64).tolist())

        uncovered_ids = np.flatnonzero(~covered_mask)
        if len(uncovered_ids) > 0:
            anchor_count = min(len(uncovered_ids), max(32, int(proposal_count) // 4))
            anchor_ids = rng.choice(uncovered_ids, size=anchor_count, replace=False)
            anchor_points = dense_points[anchor_ids]

            nearest_k = min(4, num_candidates)
            _, nearest_candidate_ids = candidate_tree.query(anchor_points, k=nearest_k)
            nearest_candidate_ids = np.asarray(nearest_candidate_ids, dtype=np.int64).reshape(-1)
            proposal_set.update(nearest_candidate_ids.tolist())

        if selected_centers:
            selected_points = np.asarray(selected_centers, dtype=np.float32)
            probe_count = min(num_candidates, max(64, int(proposal_count) // 3))
            probe_ids = rng.choice(num_candidates, size=probe_count, replace=False)
            probe_points = candidate_points[probe_ids]
            diff = probe_points[:, None, :] - selected_points[None, :, :]
            min_dist_sq = np.sum(diff * diff, axis=2).min(axis=1)
            far_ids = probe_ids[np.argsort(-min_dist_sq)[: min(64, len(probe_ids))]]
            proposal_set.update(np.asarray(far_ids, dtype=np.int64).tolist())

        proposal_ids = np.asarray(sorted(proposal_set), dtype=np.int64)

        best_score = -1
        best_center = None
        best_normal = None
        best_patch_ids = None

        for proposal_idx in np.asarray(proposal_ids, dtype=np.int64):
            center_point = candidate_points[proposal_idx]
            center_normal = candidate_normals[proposal_idx]

            if selected_centers:
                center_dists = np.linalg.norm(
                    np.asarray(selected_centers, dtype=np.float32) - center_point[None, :],
                    axis=1,
                )
                if np.min(center_dists) < float(min_center_separation):
                    continue

            patch_ids = filter_patch_ids(
                dense_points=dense_points,
                dense_normals=dense_normals,
                dense_tree=dense_tree,
                center_point=center_point,
                center_normal=center_normal,
                patch_radius=patch_radius,
                patch_thickness=patch_thickness,
                min_normal_cos=min_normal_cos,
            )
            if len(patch_ids) == 0:
                continue

            uncovered_gain = int(np.count_nonzero(~covered_mask[patch_ids]))
            if uncovered_gain > best_score:
                best_score = uncovered_gain
                best_center = center_point
                best_normal = center_normal
                best_patch_ids = patch_ids

        if best_center is None or best_score <= 0:
            rescue = pick_rescue_center(
                dense_points=dense_points,
                dense_normals=dense_normals,
                dense_tree=dense_tree,
                uncovered_mask=~covered_mask,
                selected_points=selected_centers,
                patch_radius=patch_radius,
                patch_thickness=patch_thickness,
                min_normal_cos=min_normal_cos,
                rng=rng,
            )
            if rescue is None:
                raise RuntimeError(
                    f"Could not find a valid rescue center for slot {center_slot}. "
                    "Try increasing dense surface samples or patch radius."
                )
            best_center, best_normal, best_patch_ids = rescue
            best_score = int(np.count_nonzero(~covered_mask[best_patch_ids]))

        selected_centers.append(best_center.astype(np.float32))
        selected_normals.append(best_normal.astype(np.float32))
        selected_patch_ids.append(np.asarray(best_patch_ids, dtype=np.int64))
        covered_mask[best_patch_ids] = True
        coverage_ratio = float(np.mean(covered_mask))
        coverage_progress.append(coverage_ratio)

        print(
            f"[CENTER pipeline-coverage] slot={center_slot:03d} "
            f"gain={best_score:06d} coverage={coverage_ratio:.4f}"
        )

    return {
        "centers": np.asarray(selected_centers, dtype=np.float32),
        "normals": np.asarray(selected_normals, dtype=np.float32),
        "patch_ids": selected_patch_ids,
        "coverage_progress": np.asarray(coverage_progress, dtype=np.float32),
        "covered_mask": covered_mask,
    }


def build_touch_views_from_selected_centers(
    dense_points,
    selected_centers,
    selected_normals,
    selected_patch_ids,
    num_tactile_samples,
    tactile_num_fingers,
    tactile_points_per_finger,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    num_views = int(num_tactile_samples)
    num_fingers = int(tactile_num_fingers)
    points_per_finger = int(tactile_points_per_finger)
    total_centers = num_views * num_fingers

    if len(selected_centers) != total_centers:
        raise RuntimeError(
            f"Selected center count mismatch: got {len(selected_centers)}, expected {total_centers}"
        )

    view_point_count = num_fingers * points_per_finger
    touch_points = np.zeros((num_views, view_point_count, 3), dtype=np.float32)
    touch_round_ids = np.zeros((num_views, view_point_count), dtype=np.int32)
    touch_finger_ids = np.zeros((num_views, view_point_count), dtype=np.int32)
    touch_center_ids = np.zeros((num_views, view_point_count), dtype=np.int32)
    touch_centers = np.zeros((num_views, num_fingers, 3), dtype=np.float32)
    touch_center_normals = np.zeros((num_views, num_fingers, 3), dtype=np.float32)
    touch_patch_source_counts = np.zeros((num_views, num_fingers), dtype=np.int32)

    for view_idx in range(num_views):
        for finger_idx in range(num_fingers):
            global_idx = view_idx * num_fingers + finger_idx
            patch_ids = selected_patch_ids[global_idx]
            patch_points = sample_patch_points_from_ids(
                dense_points=dense_points,
                patch_ids=patch_ids,
                points_per_touch=points_per_finger,
                rng=rng,
            )

            start = finger_idx * points_per_finger
            end = start + points_per_finger

            touch_points[view_idx, start:end] = patch_points
            touch_finger_ids[view_idx, start:end] = finger_idx
            touch_center_ids[view_idx, start:end] = finger_idx
            touch_centers[view_idx, finger_idx] = selected_centers[global_idx]
            touch_center_normals[view_idx, finger_idx] = selected_normals[global_idx]
            touch_patch_source_counts[view_idx, finger_idx] = int(len(patch_ids))

    return {
        "touch_points": touch_points,
        "touch_round_ids": touch_round_ids,
        "touch_finger_ids": touch_finger_ids,
        "touch_center_ids": touch_center_ids,
        "touch_centers": touch_centers,
        "touch_center_normals": touch_center_normals,
        "touch_patch_source_counts": touch_patch_source_counts,
    }


def process_single_obj_to_pipeline_coverage_npz(
    obj_path,
    out_path,
    num_tactile_samples=10,
    tactile_num_fingers=10,
    tactile_points_per_finger=3000,
    tactile_patch_radius_ratio=0.10,
    tactile_patch_thickness_ratio=0.035,
    tactile_min_center_separation_ratio=0.065,
    dense_surface_sample_n=120000,
    candidate_touch_samples=6000,
    patch_min_normal_cos=0.0,
    num_surface_points=235000,
    num_query_points=250000,
):
    print("\n==================================================")
    print("[PROCESS pipeline-style coverage-aware]", obj_path)
    print("==================================================")

    mesh_name = os.path.splitext(os.path.basename(obj_path))[0]

    mesh = trimesh.load(obj_path, force="mesh")
    mesh.process(validate=True)
    mesh = scale_to_unit_sphere(mesh)

    surface_points, surface_normals = sample_surface_points_for_storage(
        mesh,
        num_surface_points=num_surface_points,
    )

    scene = build_raycast_scene(mesh)

    print("[INFO] sampling query points near the surface ...")
    query_points = sample_query_points_near_surface(
        surface_points=surface_points,
        number_of_points=num_query_points,
    )

    if mesh.is_watertight:
        print("[INFO] computing query_sdf using RaycastingScene occupancy sign (watertight mesh) ...")
    else:
        print("[INFO] computing query_sdf using unsigned distance + near-surface sign fallback (non-watertight mesh) ...")

    query_sdf = compute_query_sdf_with_raycasting(
        scene=scene,
        query_points=query_points,
        mesh_is_watertight=mesh.is_watertight,
        surface_points=surface_points,
        surface_normals=surface_normals,
        occupancy_nsamples=11,
        near_surface_sign_band=0.01,
    )

    print("[INFO] sampling dense surface points for pipeline-style tactile planning ...")
    dense_points, dense_normals = sample_dense_surface_points(
        mesh=mesh,
        sample_count=dense_surface_sample_n,
    )
    diag = compute_bbox_diag(dense_points)

    patch_radius = float(tactile_patch_radius_ratio) * diag
    patch_thickness = float(tactile_patch_thickness_ratio) * diag
    min_center_separation = float(tactile_min_center_separation_ratio) * diag
    total_centers = int(num_tactile_samples) * int(tactile_num_fingers)

    print(
        "[INFO] planning tactile patches with global greedy coverage "
        f"(total_centers={total_centers}, patch_radius={patch_radius:.6f}) ..."
    )

    planning_rng = np.random.default_rng()
    candidate_points, candidate_normals = sample_candidate_touch_targets(
        mesh=mesh,
        candidate_count=max(int(candidate_touch_samples), total_centers * 8),
        seed=int(planning_rng.integers(0, 2**31 - 1)),
    )

    selected = greedy_select_touch_targets(
        dense_points=dense_points,
        dense_normals=dense_normals,
        candidate_points=candidate_points,
        candidate_normals=candidate_normals,
        total_centers=total_centers,
        patch_radius=patch_radius,
        patch_thickness=patch_thickness,
        min_center_separation=min_center_separation,
        min_normal_cos=patch_min_normal_cos,
        proposal_count=256,
        rng=planning_rng,
    )

    touch_data = build_touch_views_from_selected_centers(
        dense_points=dense_points,
        selected_centers=selected["centers"],
        selected_normals=selected["normals"],
        selected_patch_ids=selected["patch_ids"],
        num_tactile_samples=num_tactile_samples,
        tactile_num_fingers=tactile_num_fingers,
        tactile_points_per_finger=tactile_points_per_finger,
        rng=np.random.default_rng(),
    )

    coverage_progress = selected["coverage_progress"]
    view_coverages = coverage_progress.reshape(int(num_tactile_samples), int(tactile_num_fingers))[:, -1]
    final_coverage_ratio = float(coverage_progress[-1]) if len(coverage_progress) > 0 else 0.0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    np.savez_compressed(
        out_path,
        surface_points=surface_points,
        surface_normals=surface_normals,
        query_points=query_points.astype(np.float32),
        query_sdf=query_sdf.astype(np.float32),
        touch_points=touch_data["touch_points"],
        touch_round_ids=touch_data["touch_round_ids"],
        touch_finger_ids=touch_data["touch_finger_ids"],
        touch_center_ids=touch_data["touch_center_ids"],
        touch_centers=touch_data["touch_centers"],
        touch_center_normals=touch_data["touch_center_normals"],
        touch_patch_source_counts=touch_data["touch_patch_source_counts"],
        planning_surface_coverage_ratio=np.array(final_coverage_ratio, dtype=np.float32),
        planning_view_coverage_ratio=view_coverages.astype(np.float32),
        mesh_name=np.array(mesh_name),
        num_tactile_samples=np.array(num_tactile_samples, dtype=np.int32),
        tactile_num_fingers=np.array(tactile_num_fingers, dtype=np.int32),
    )

    print("[SAVED]", out_path)
    print("surface_points                 :", surface_points.shape)
    print("query_points                   :", query_points.shape)
    print("query_sdf                      :", query_sdf.shape)
    print("touch_points                   :", touch_data["touch_points"].shape)
    print("touch_centers                  :", touch_data["touch_centers"].shape)
    print(f"planning_surface_coverage_ratio: {final_coverage_ratio:.4f}")


def process_shapenetcore_all_categories_pipeline_coverage_onefolder(
    root_dir,
    category_names=None,
    max_objects_per_category=None,
    num_tactile_samples=10,
    tactile_num_fingers=10,
    tactile_points_per_finger=3000,
    tactile_patch_radius_ratio=0.10,
    tactile_patch_thickness_ratio=0.035,
    tactile_min_center_separation_ratio=0.065,
    dense_surface_sample_n=120000,
    candidate_touch_samples=6000,
    output_folder_name="tactistruct_npz_shapenet_pipeline_coverage_onefolder",
):
    category_dirs = list(iter_category_dirs(root_dir, category_names=category_names))

    if not category_dirs:
        print("[WARN] no category folders found under:", root_dir)
        return

    out_dir = os.path.join(root_dir, output_folder_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] found {len(category_dirs)} category folders under root.")
    print(f"[INFO] flat output dir: {out_dir}")

    for category_dir in category_dirs:
        category_name = os.path.basename(category_dir)
        obj_paths = find_shapenet_obj_files(
            category_dir,
            max_objects=max_objects_per_category,
        )

        print(f"\n########## Processing category: {category_name} ##########")
        print(f"[INFO] found {len(obj_paths)} model_normalized.obj files")

        if not obj_paths:
            continue

        for obj_path in obj_paths:
            out_path = build_flat_output_path(
                obj_path=obj_path,
                root_dir=root_dir,
                output_folder_name=output_folder_name,
            )

            if os.path.exists(out_path):
                print("[SKIP exists]", out_path)
                continue

            try:
                process_single_obj_to_pipeline_coverage_npz(
                    obj_path=obj_path,
                    out_path=out_path,
                    num_tactile_samples=num_tactile_samples,
                    tactile_num_fingers=tactile_num_fingers,
                    tactile_points_per_finger=tactile_points_per_finger,
                    tactile_patch_radius_ratio=tactile_patch_radius_ratio,
                    tactile_patch_thickness_ratio=tactile_patch_thickness_ratio,
                    tactile_min_center_separation_ratio=tactile_min_center_separation_ratio,
                    dense_surface_sample_n=dense_surface_sample_n,
                    candidate_touch_samples=candidate_touch_samples,
                )
            except Exception as e:
                print("[FAILED]", obj_path)
                print("Error:", e)


if __name__ == "__main__":
    root_dir = r"C:/Users/wudaw/Downloads/ShapeNetCore/ShapeNetCore"

    process_shapenetcore_all_categories_pipeline_coverage_onefolder(
        root_dir=root_dir,
        category_names=None,
        max_objects_per_category=275,
        num_tactile_samples=10,
        tactile_num_fingers=10,
        tactile_points_per_finger=3000,
        tactile_patch_radius_ratio=0.10,
        tactile_patch_thickness_ratio=0.035,
        tactile_min_center_separation_ratio=0.065,
        dense_surface_sample_n=120000,
        candidate_touch_samples=6000,
        output_folder_name="tactistruct_npz_shapenet_pipeline_coverage_onefolder",
    )

    print("\nAll done.")
