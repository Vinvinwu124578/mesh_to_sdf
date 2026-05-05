import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from SDF_batch_sampling_new_paper_idea import (
    bbox_diag,
    build_global_reachable_mask,
    build_raycast_scene,
    compute_patch_overlap_ratio,
    compute_query_sdf_with_raycasting,
    compute_soft_contact_probability,
    extract_patch_candidates_fixed_center,
    raycast_outer_hits,
    resolve_multi_finger_candidates,
    sample_from_mesh,
    sample_query_points_near_surface,
    sample_surface_points_for_storage,
    scale_to_unit_sphere,
)


def build_tactile_planning_context(
    mesh,
    surface_sample_n=200000,
    num_rays=20000,
    beam_radius=0.1,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    spc = sample_from_mesh(mesh, n=surface_sample_n)
    diag = bbox_diag(spc.points)

    candidate_ids, hit_points = raycast_outer_hits(
        mesh=mesh,
        points=spc.points,
        num_rays=num_rays,
        rng=rng,
    )

    reachable_mask = build_global_reachable_mask(
        spc.points,
        hit_points,
        beam_radius=beam_radius,
    )

    point_tree = cKDTree(spc.points)
    scene = build_raycast_scene(mesh)

    return {
        "spc": spc,
        "diag": diag,
        "candidate_ids": np.asarray(candidate_ids, dtype=np.int64),
        "reachable_mask": reachable_mask.astype(bool),
        "point_tree": point_tree,
        "scene": scene,
    }


def score_center_approximate(
    points,
    point_tree,
    reachable_mask,
    covered_mask,
    center_idx,
    radius,
):
    local_ids = point_tree.query_ball_point(points[center_idx], r=float(radius))
    if not local_ids:
        return 0

    local_ids = np.asarray(local_ids, dtype=np.int64)
    local_ids = local_ids[reachable_mask[local_ids]]
    if len(local_ids) == 0:
        return 0

    return int(np.count_nonzero(~covered_mask[local_ids]))


def build_mask_from_ids(num_points, ids):
    mask = np.zeros(num_points, dtype=bool)
    if len(ids) > 0:
        mask[np.asarray(ids, dtype=np.int64)] = True
    return mask


def select_centers_for_coverage(
    scene,
    points,
    normals,
    point_tree,
    candidate_ids,
    reachable_mask,
    covered_mask,
    radius,
    thickness,
    num_centers=10,
    min_center_dist=None,
    max_patch_overlap=0.15,
    approx_eval_count=192,
    exact_eval_count=24,
    fallback_trials=4096,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    if len(candidate_ids) == 0:
        raise RuntimeError("candidate_ids is empty in select_centers_for_coverage.")

    if min_center_dist is None:
        min_center_dist = 1.1 * radius

    centers = []
    accepted_masks = []
    local_covered_mask = covered_mask.copy()

    for center_slot in range(num_centers):
        preferred_ids = candidate_ids[~local_covered_mask[candidate_ids]]
        if len(preferred_ids) == 0:
            preferred_ids = candidate_ids

        if len(preferred_ids) > approx_eval_count:
            proposal_ids = rng.choice(preferred_ids, size=approx_eval_count, replace=False)
        else:
            proposal_ids = preferred_ids

        approx_ranked = []
        for idx in np.asarray(proposal_ids, dtype=np.int64):
            if any(np.linalg.norm(points[idx] - points[c]) < min_center_dist for c in centers):
                continue

            approx_score = score_center_approximate(
                points=points,
                point_tree=point_tree,
                reachable_mask=reachable_mask,
                covered_mask=local_covered_mask,
                center_idx=int(idx),
                radius=radius,
            )
            if approx_score > 0:
                approx_ranked.append((approx_score, int(idx)))

        approx_ranked.sort(reverse=True)
        exact_candidates = [idx for _, idx in approx_ranked[:exact_eval_count]]

        best_choice = None
        best_score = -1
        best_mask = None
        best_info = None

        for idx in exact_candidates:
            patch_ids, info = extract_patch_candidates_fixed_center(
                scene=scene,
                points=points,
                normals=normals,
                reachable_mask=reachable_mask,
                center_idx=idx,
                radius=radius,
                thickness=thickness,
            )
            if len(patch_ids) == 0:
                continue

            patch_mask = build_mask_from_ids(len(points), patch_ids)
            overlap_too_large = any(
                compute_patch_overlap_ratio(patch_mask, prev_mask) > max_patch_overlap
                for prev_mask in accepted_masks
            )
            if overlap_too_large:
                continue

            uncovered_gain = int(np.count_nonzero(patch_mask & ~local_covered_mask))
            if uncovered_gain > best_score:
                best_choice = int(idx)
                best_score = uncovered_gain
                best_mask = patch_mask
                best_info = info

        if best_choice is None:
            trials = 0
            while trials < fallback_trials:
                trials += 1
                idx = int(rng.choice(candidate_ids))

                if any(np.linalg.norm(points[idx] - points[c]) < min_center_dist for c in centers):
                    continue

                patch_ids, info = extract_patch_candidates_fixed_center(
                    scene=scene,
                    points=points,
                    normals=normals,
                    reachable_mask=reachable_mask,
                    center_idx=idx,
                    radius=radius,
                    thickness=thickness,
                )
                if len(patch_ids) == 0:
                    continue

                patch_mask = build_mask_from_ids(len(points), patch_ids)
                overlap_too_large = any(
                    compute_patch_overlap_ratio(patch_mask, prev_mask) > max_patch_overlap
                    for prev_mask in accepted_masks
                )
                if overlap_too_large:
                    continue

                best_choice = idx
                best_mask = patch_mask
                best_info = info
                best_score = int(np.count_nonzero(patch_mask & ~local_covered_mask))
                break

        if best_choice is None:
            raise RuntimeError(
                f"Only found {len(centers)} coverage-aware centers, need {num_centers}. "
                f"Try increasing surface_sample_n / num_rays, or reducing start_ratio."
            )

        centers.append(best_choice)
        accepted_masks.append(best_mask)
        local_covered_mask |= best_mask

        print(
            f"[CENTER coverage-aware] accept center {center_slot}: idx={best_choice}, "
            f"new_cover={best_score}, visible={best_info.get('n_visible', 0)}, "
            f"source={best_info.get('source', 'unknown')}"
        )

    return np.asarray(centers, dtype=np.int64), accepted_masks, local_covered_mask


def tactile_sampling_round_with_fixed_centers(
    spc,
    scene,
    reachable_mask,
    center_ids,
    radius,
    thickness,
    points_per_finger=3000,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    points = spc.points
    normals = spc.normals

    candidate_ids_per_finger = []
    info_per_finger = []

    for fid, center_idx in enumerate(center_ids):
        candidate_ids, info = extract_patch_candidates_fixed_center(
            scene=scene,
            points=points,
            normals=normals,
            reachable_mask=reachable_mask,
            center_idx=center_idx,
            radius=radius,
            thickness=thickness,
        )
        candidate_ids_per_finger.append(candidate_ids)
        info_per_finger.append(info)

    exclusive_ids_per_finger = resolve_multi_finger_candidates(
        points=points,
        center_ids=center_ids,
        candidate_ids_per_finger=candidate_ids_per_finger,
    )

    all_pts = []
    all_fids = []
    all_center_ids = []

    for fid, center_idx in enumerate(center_ids):
        candidate_ids = exclusive_ids_per_finger[fid]
        if len(candidate_ids) == 0:
            candidate_ids = np.asarray([center_idx], dtype=np.int64)

        prob = compute_soft_contact_probability(
            points=points,
            normals=normals,
            center_idx=center_idx,
            candidate_ids=candidate_ids,
            radius=radius,
            cross_surface_gain=0.35,
            edge_neighbor_ratio=0.35,
        )

        choose = rng.choice(
            candidate_ids,
            size=points_per_finger,
            replace=len(candidate_ids) < points_per_finger,
            p=prob,
        )

        all_pts.append(points[choose])
        all_fids.append(np.full(points_per_finger, fid, dtype=np.int32))
        all_center_ids.append(np.full(points_per_finger, center_idx, dtype=np.int32))

        print(
            f"[Finger {fid}] source={info_per_finger[fid]['source']} | "
            f"raw={len(candidate_ids_per_finger[fid])} | "
            f"exclusive={len(candidate_ids)} | sampled={points_per_finger}"
        )

    pts = np.vstack(all_pts).astype(np.float32)
    fids = np.concatenate(all_fids).astype(np.int32)
    cids = np.concatenate(all_center_ids).astype(np.int32)
    return pts, fids, cids


def generate_tactile_touch_points_coverage_aware(
    planning_context,
    covered_mask,
    rounds=5,
    start_ratio=0.12,
    end_ratio=0.03,
    thickness_ratio=0.01,
    points_per_finger=3000,
    num_fingers=10,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    spc = planning_context["spc"]
    scene = planning_context["scene"]
    candidate_ids = planning_context["candidate_ids"]
    reachable_mask = planning_context["reachable_mask"]
    point_tree = planning_context["point_tree"]
    diag = planning_context["diag"]

    largest_radius = start_ratio * diag
    thickness = thickness_ratio * diag

    center_ids, center_masks, updated_covered_mask = select_centers_for_coverage(
        scene=scene,
        points=spc.points,
        normals=spc.normals,
        point_tree=point_tree,
        candidate_ids=candidate_ids,
        reachable_mask=reachable_mask,
        covered_mask=covered_mask,
        radius=largest_radius,
        thickness=thickness,
        num_centers=num_fingers,
        min_center_dist=1.1 * largest_radius,
        max_patch_overlap=0.15,
        approx_eval_count=192,
        exact_eval_count=24,
        rng=rng,
    )

    print("[INFO] coverage-aware center ids:", center_ids.tolist())

    radius_schedule = np.linspace(start_ratio, end_ratio, rounds)

    round_points = []
    round_ids = []
    finger_ids = []
    center_ids_per_point = []

    for rid, ratio in enumerate(radius_schedule):
        radius = float(ratio) * diag
        print(f"\n[ROUND {rid}] radius_ratio={ratio:.4f}, radius={radius:.6f}")

        pts, fids, cids = tactile_sampling_round_with_fixed_centers(
            spc=spc,
            scene=scene,
            reachable_mask=reachable_mask,
            center_ids=center_ids,
            radius=radius,
            thickness=thickness,
            points_per_finger=points_per_finger,
            rng=rng,
        )

        round_points.append(pts)
        round_ids.append(np.full(len(pts), rid, dtype=np.int32))
        finger_ids.append(fids)
        center_ids_per_point.append(cids)

    coverage_mask = np.zeros(len(spc.points), dtype=bool)
    for patch_mask in center_masks:
        coverage_mask |= patch_mask

    touch_points = np.vstack(round_points).astype(np.float32)
    touch_round_ids = np.concatenate(round_ids).astype(np.int32)
    touch_finger_ids = np.concatenate(finger_ids).astype(np.int32)
    touch_center_ids = np.concatenate(center_ids_per_point).astype(np.int32)
    touch_centers = spc.points[center_ids].astype(np.float32)

    expected_n = rounds * num_fingers * points_per_finger
    if len(touch_points) != expected_n:
        raise RuntimeError(
            f"Tactile point count mismatch: got {len(touch_points)}, expected {expected_n}"
        )

    return {
        "touch_points": touch_points,
        "touch_round_ids": touch_round_ids,
        "touch_finger_ids": touch_finger_ids,
        "touch_center_ids": touch_center_ids,
        "touch_centers": touch_centers,
        "coverage_mask": coverage_mask,
        "updated_covered_mask": updated_covered_mask,
    }


def process_single_obj_to_merged_npz_coverage_aware(
    obj_path,
    out_path,
    num_tactile_samples=10,
    tactile_surface_sample_n=200000,
    tactile_num_rays=20000,
    tactile_beam_radius=0.1,
    tactile_rounds=5,
    tactile_start_ratio=0.12,
    tactile_end_ratio=0.03,
    tactile_thickness_ratio=0.01,
    tactile_points_per_finger=3000,
    tactile_num_fingers=10,
    num_surface_points=235000,
    num_query_points=250000,
):
    print("\n==================================================")
    print("[PROCESS coverage-aware]", obj_path)
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

    planning_rng = np.random.default_rng()
    planning_context = build_tactile_planning_context(
        mesh=mesh,
        surface_sample_n=tactile_surface_sample_n,
        num_rays=tactile_num_rays,
        beam_radius=tactile_beam_radius,
        rng=planning_rng,
    )

    global_covered_mask = np.zeros(len(planning_context["spc"].points), dtype=bool)

    touch_points_all = []
    touch_round_ids_all = []
    touch_finger_ids_all = []
    touch_center_ids_all = []
    touch_centers_all = []

    for sample_idx in range(num_tactile_samples):
        print(f"[INFO] tactile sample {sample_idx + 1}/{num_tactile_samples}")

        rng = np.random.default_rng()

        tactile_data = generate_tactile_touch_points_coverage_aware(
            planning_context=planning_context,
            covered_mask=global_covered_mask,
            rounds=tactile_rounds,
            start_ratio=tactile_start_ratio,
            end_ratio=tactile_end_ratio,
            thickness_ratio=tactile_thickness_ratio,
            points_per_finger=tactile_points_per_finger,
            num_fingers=tactile_num_fingers,
            rng=rng,
        )

        global_covered_mask = tactile_data["updated_covered_mask"].copy()
        current_cover = float(np.mean(global_covered_mask))
        print(f"[INFO] planning surface covered so far: {current_cover:.4f}")

        touch_points_all.append(tactile_data["touch_points"])
        touch_round_ids_all.append(tactile_data["touch_round_ids"])
        touch_finger_ids_all.append(tactile_data["touch_finger_ids"])
        touch_center_ids_all.append(tactile_data["touch_center_ids"])
        touch_centers_all.append(tactile_data["touch_centers"])

    touch_points_all = np.stack(touch_points_all, axis=0).astype(np.float32)
    touch_round_ids_all = np.stack(touch_round_ids_all, axis=0).astype(np.int32)
    touch_finger_ids_all = np.stack(touch_finger_ids_all, axis=0).astype(np.int32)
    touch_center_ids_all = np.stack(touch_center_ids_all, axis=0).astype(np.int32)
    touch_centers_all = np.stack(touch_centers_all, axis=0).astype(np.float32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    np.savez_compressed(
        out_path,
        surface_points=surface_points,
        surface_normals=surface_normals,
        query_points=query_points.astype(np.float32),
        query_sdf=query_sdf.astype(np.float32),
        touch_points=touch_points_all,
        touch_round_ids=touch_round_ids_all,
        touch_finger_ids=touch_finger_ids_all,
        touch_center_ids=touch_center_ids_all,
        touch_centers=touch_centers_all,
        mesh_name=np.array(mesh_name),
        num_tactile_samples=np.array(num_tactile_samples, dtype=np.int32),
    )

    print("[SAVED]", out_path)
    print("surface_points   :", surface_points.shape)
    print("surface_normals  :", surface_normals.shape)
    print("query_points     :", query_points.shape)
    print("query_sdf        :", query_sdf.shape)
    print("touch_points     :", touch_points_all.shape)
    print("touch_round_ids  :", touch_round_ids_all.shape)
