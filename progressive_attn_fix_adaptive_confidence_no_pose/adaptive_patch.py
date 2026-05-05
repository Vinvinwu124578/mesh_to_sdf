from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def object_seed(base_seed: int, obj_path: str) -> int:
    digest = hashlib.sha1(obj_path.encode("utf-8")).hexdigest()[:8]
    return int(base_seed) + int(digest, 16)


def build_flat_output_path(obj_path: str, root_dir: str, output_folder_name: str) -> str:
    rel_path = Path(obj_path).relative_to(root_dir)
    rel_without_ext = rel_path.with_suffix("")
    safe_name = str(rel_without_ext).replace("\\", "__").replace("/", "__").replace(":", "_")
    return str(Path(root_dir) / output_folder_name / f"{safe_name}.npz")


def build_asset_export_path(obj_path: str, root_dir: str, asset_folder_name: str) -> str:
    rel_path = Path(obj_path).relative_to(root_dir)
    rel_without_ext = rel_path.with_suffix("")
    safe_name = str(rel_without_ext).replace("\\", "__").replace("/", "__").replace(":", "_")
    return str(Path(root_dir) / asset_folder_name / f"{safe_name}__normalized.stl")


def subsample_or_repeat(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0:
        raise ValueError("count must be positive.")
    if len(points) == 0:
        feature_dim = int(points.shape[-1]) if points.ndim == 2 else 3
        return np.zeros((count, feature_dim), dtype=np.float32)
    if len(points) >= count:
        indices = rng.choice(len(points), size=count, replace=False)
        return points[indices].astype(np.float32)
    extra = rng.choice(len(points), size=count - len(points), replace=True)
    return np.concatenate([points, points[extra]], axis=0).astype(np.float32)


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


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (vec / norm).astype(np.float32)


def build_tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normalize_vector(normal)
    reference = np.array([0.0, 0.0, 1.0], dtype=np.float32) if abs(float(normal[2])) < 0.95 else np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    tangent_u = normalize_vector(np.cross(normal, reference))
    tangent_v = normalize_vector(np.cross(normal, tangent_u))
    return tangent_u, tangent_v


def keep_plane_cluster_near_center(signed_depth: np.ndarray, gap_threshold: float) -> tuple[np.ndarray, bool]:
    if signed_depth.size <= 1 or gap_threshold <= 0.0:
        return np.ones(signed_depth.shape[0], dtype=bool), False

    order = np.argsort(signed_depth)
    sorted_depth = signed_depth[order]
    gaps = np.diff(sorted_depth)
    split_positions = np.where(gaps > gap_threshold)[0]
    if split_positions.size == 0:
        return np.ones(signed_depth.shape[0], dtype=bool), False

    sorted_cluster_ids = np.zeros(sorted_depth.shape[0], dtype=np.int32)
    cluster_id = 0
    for split_position in split_positions:
        cluster_id += 1
        sorted_cluster_ids[split_position + 1 :] = cluster_id

    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(order.shape[0])
    cluster_ids = sorted_cluster_ids[inverse_order]
    center_index = int(np.argmin(np.abs(signed_depth)))
    target_cluster = int(cluster_ids[center_index])
    return cluster_ids == target_cluster, True


def query_normals(tree: cKDTree, surface_normals: np.ndarray, points: np.ndarray) -> np.ndarray:
    _, indices = tree.query(points.astype(np.float32), k=1)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    return surface_normals[indices].astype(np.float32)


def compute_patch_axes(tangent_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if tangent_xy.shape[0] <= 1:
        return tangent_xy.astype(np.float32), np.eye(2, dtype=np.float32)
    covariance = tangent_xy.T @ tangent_xy / float(max(tangent_xy.shape[0], 1))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance.astype(np.float64))
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order].astype(np.float32)
    return (tangent_xy @ axes).astype(np.float32), axes


def estimate_patch_statistics(
    raw_patch_points: np.ndarray,
    raw_patch_normals: np.ndarray,
    center_point: np.ndarray,
    center_normal: np.ndarray,
    target_point: np.ndarray,
    target_normal: np.ndarray,
    contact_point: np.ndarray,
    contact_normal: np.ndarray,
    adaptive_minor_radius_floor_ratio: float,
    adaptive_minor_radius_scale: float,
    adaptive_major_radius_scale: float,
    adaptive_plane_gap_ratio: float,
    adaptive_min_points_per_patch: int,
    confidence_floor: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | bool]]:
    center_point = np.asarray(center_point, dtype=np.float32)
    center_normal = normalize_vector(center_normal)
    target_point = np.asarray(target_point, dtype=np.float32)
    target_normal = normalize_vector(target_normal)
    contact_point = np.asarray(contact_point, dtype=np.float32)
    contact_normal = normalize_vector(contact_normal)

    tangent_u, tangent_v = build_tangent_basis(center_normal)
    offsets = raw_patch_points - center_point[None, :]
    signed_depth = offsets @ center_normal
    tangent_xy = np.stack([offsets @ tangent_u, offsets @ tangent_v], axis=-1).astype(np.float32)

    radial_dist = np.linalg.norm(tangent_xy, axis=-1)
    base_radius = float(np.quantile(radial_dist, 0.90)) if radial_dist.size > 0 else 1e-3
    base_radius = max(base_radius, 1e-3)

    plane_gap_threshold = max(float(adaptive_plane_gap_ratio) * base_radius, 1e-4)
    plane_mask, plane_split_flag = keep_plane_cluster_near_center(signed_depth, plane_gap_threshold)
    plane_points = raw_patch_points[plane_mask]
    plane_normals = raw_patch_normals[plane_mask]
    plane_xy = tangent_xy[plane_mask]
    plane_depth = signed_depth[plane_mask]
    if plane_points.shape[0] == 0:
        plane_points = raw_patch_points
        plane_normals = raw_patch_normals
        plane_xy = tangent_xy
        plane_depth = signed_depth
        plane_split_flag = False

    projected_xy, _ = compute_patch_axes(plane_xy)
    major_coord = projected_xy[:, 0]
    minor_coord = projected_xy[:, 1]
    major_extent = float(np.quantile(np.abs(major_coord), 0.95)) if major_coord.size > 0 else base_radius
    minor_extent = float(np.quantile(np.abs(minor_coord), 0.95)) if minor_coord.size > 0 else base_radius
    major_extent = max(major_extent, 1e-4)
    minor_extent = max(minor_extent, 1e-4)

    major_radius = min(max(base_radius * 0.5, major_extent * float(adaptive_major_radius_scale)), base_radius)
    minor_floor = max(float(adaptive_minor_radius_floor_ratio) * base_radius, 1e-4)
    minor_radius = min(max(minor_extent * float(adaptive_minor_radius_scale), minor_floor), major_radius)

    ellipse_score = (major_coord / max(major_radius, 1e-6)) ** 2 + (minor_coord / max(minor_radius, 1e-6)) ** 2
    ellipse_mask = ellipse_score <= 1.0
    filtered_points = plane_points[ellipse_mask]
    filtered_normals = plane_normals[ellipse_mask]
    filtered_score = ellipse_score[ellipse_mask]
    if filtered_points.shape[0] < int(adaptive_min_points_per_patch):
        fallback_count = min(max(int(adaptive_min_points_per_patch), 1), plane_points.shape[0])
        fallback_ids = np.argsort(ellipse_score)[:fallback_count]
        filtered_points = plane_points[fallback_ids]
        filtered_normals = plane_normals[fallback_ids]
        filtered_score = ellipse_score[fallback_ids]

    cos_center = np.clip(filtered_normals @ center_normal, -1.0, 1.0)
    normal_variance = float(np.clip(1.0 - np.mean(np.abs(cos_center)), 0.0, 1.0))
    anisotropy = float(np.clip(minor_extent / max(major_extent, 1e-6), 0.0, 1.0))
    edge_narrowness = float(np.clip(1.0 - anisotropy, 0.0, 1.0))
    target_contact_offset = float(np.linalg.norm(contact_point - target_point))
    target_contact_offset_ratio = float(np.clip(target_contact_offset / max(base_radius, 1e-6), 0.0, 4.0))
    normal_agreement = float(np.clip(np.dot(target_normal, contact_normal), -1.0, 1.0))
    reachability_margin = float(np.clip(0.5 * (normal_agreement + 1.0), 0.0, 1.0))
    reachability_margin = float(np.clip(reachability_margin * np.exp(-target_contact_offset_ratio), 0.0, 1.0))

    edge_score = (
        0.34 * edge_narrowness
        + 0.22 * normal_variance
        + 0.18 * float(np.clip(target_contact_offset_ratio / 1.2, 0.0, 1.0))
        + 0.14 * float(plane_split_flag)
        + 0.12 * (1.0 - reachability_margin)
    )
    edge_score = float(np.clip(edge_score, 0.0, 1.0))
    patch_confidence = float(np.clip(1.0 - edge_score, float(confidence_floor), 1.0))
    band_width = float(
        np.clip(
            0.004 + 0.015 * edge_score + 0.10 * max(major_radius - minor_radius, 0.0) + 0.01 * normal_variance,
            0.004,
            0.045,
        )
    )

    stats = {
        "patch_confidence": patch_confidence,
        "patch_edge_score": edge_score,
        "patch_band_width": band_width,
        "patch_major_extent": float(major_extent),
        "patch_minor_extent": float(minor_extent),
        "patch_normal_variance": normal_variance,
        "patch_plane_split_flag": bool(plane_split_flag),
        "patch_reachability_margin": reachability_margin,
        "patch_target_contact_offset": target_contact_offset,
        "patch_target_contact_offset_ratio": target_contact_offset_ratio,
        "patch_adaptive_major_radius": float(major_radius),
        "patch_adaptive_minor_radius": float(minor_radius),
        "patch_depth_spread": float(np.quantile(np.abs(plane_depth), 0.95)) if plane_depth.size > 0 else 0.0,
        "patch_filter_score_mean": float(filtered_score.mean()) if filtered_score.size > 0 else 0.0,
    }
    return filtered_points.astype(np.float32), filtered_normals.astype(np.float32), stats


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


def structure_single_view_adaptive(
    touch_points: np.ndarray,
    round_ids: np.ndarray,
    finger_ids: np.ndarray,
    touch_centers: np.ndarray,
    touch_center_normals: np.ndarray,
    touch_target_points: np.ndarray,
    touch_target_normals: np.ndarray,
    touch_contact_points: np.ndarray,
    touch_contact_normals: np.ndarray,
    surface_tree: cKDTree,
    surface_normals: np.ndarray,
    points_per_patch: int,
    rng: np.random.Generator,
    adaptive_minor_radius_floor_ratio: float,
    adaptive_minor_radius_scale: float,
    adaptive_major_radius_scale: float,
    adaptive_plane_gap_ratio: float,
    adaptive_min_points_per_patch: int,
    confidence_floor: float,
) -> dict[str, np.ndarray]:
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
    patch_normals = np.zeros((num_fingers, num_rounds, points_per_patch, 3), dtype=np.float32)
    patch_point_confidence = np.zeros((num_fingers, num_rounds, points_per_patch), dtype=np.float32)
    patch_point_band_width = np.zeros((num_fingers, num_rounds, points_per_patch), dtype=np.float32)
    patch_mask = np.zeros((num_fingers, num_rounds), dtype=bool)
    patch_counts = np.zeros((num_fingers, num_rounds), dtype=np.int32)
    patch_radii = np.zeros((num_fingers, num_rounds), dtype=np.float32)
    patch_confidence = np.zeros((num_fingers, num_rounds), dtype=np.float32)
    patch_edge_score = np.zeros((num_fingers, num_rounds), dtype=np.float32)
    patch_band_width = np.zeros((num_fingers, num_rounds), dtype=np.float32)
    patch_major_extent = np.zeros((num_fingers, num_rounds), dtype=np.float32)
    patch_minor_extent = np.zeros((num_fingers, num_rounds), dtype=np.float32)
    patch_normal_variance = np.zeros((num_fingers, num_rounds), dtype=np.float32)
    patch_plane_split_flag = np.zeros((num_fingers, num_rounds), dtype=bool)
    patch_reachability_margin = np.zeros((num_fingers, num_rounds), dtype=np.float32)
    patch_target_contact_offset = np.zeros((num_fingers, num_rounds), dtype=np.float32)
    patch_target_contact_offset_ratio = np.zeros((num_fingers, num_rounds), dtype=np.float32)
    patch_adaptive_major_radius = np.zeros((num_fingers, num_rounds), dtype=np.float32)
    patch_adaptive_minor_radius = np.zeros((num_fingers, num_rounds), dtype=np.float32)

    for finger_index in range(num_fingers):
        center_point = centers[finger_index]
        center_normal = (
            touch_center_normals[finger_index]
            if finger_index < touch_center_normals.shape[0]
            else query_normals(surface_tree, surface_normals, center_point[None, :])[0]
        )
        target_point = touch_target_points[finger_index] if finger_index < touch_target_points.shape[0] else center_point
        target_normal = touch_target_normals[finger_index] if finger_index < touch_target_normals.shape[0] else center_normal
        contact_point = touch_contact_points[finger_index] if finger_index < touch_contact_points.shape[0] else target_point
        contact_normal = (
            touch_contact_normals[finger_index] if finger_index < touch_contact_normals.shape[0] else target_normal
        )
        for round_index in range(num_rounds):
            patch_selector = (finger_ids == finger_index) & (round_ids == round_index)
            raw_patch_points = touch_points[patch_selector]
            if len(raw_patch_points) == 0:
                patch_points[finger_index, round_index] = center_point[None, :]
                patch_normals[finger_index, round_index] = normalize_vector(center_normal)[None, :]
                patch_point_confidence[finger_index, round_index] = float(confidence_floor)
                patch_point_band_width[finger_index, round_index] = 0.03
                continue

            raw_patch_normals = query_normals(surface_tree, surface_normals, raw_patch_points)
            adaptive_points, adaptive_normals, stats = estimate_patch_statistics(
                raw_patch_points=raw_patch_points,
                raw_patch_normals=raw_patch_normals,
                center_point=center_point,
                center_normal=center_normal,
                target_point=target_point,
                target_normal=target_normal,
                contact_point=contact_point,
                contact_normal=contact_normal,
                adaptive_minor_radius_floor_ratio=adaptive_minor_radius_floor_ratio,
                adaptive_minor_radius_scale=adaptive_minor_radius_scale,
                adaptive_major_radius_scale=adaptive_major_radius_scale,
                adaptive_plane_gap_ratio=adaptive_plane_gap_ratio,
                adaptive_min_points_per_patch=adaptive_min_points_per_patch,
                confidence_floor=confidence_floor,
            )

            patch_mask[finger_index, round_index] = True
            patch_counts[finger_index, round_index] = int(adaptive_points.shape[0])
            patch_radii[finger_index, round_index] = max(
                float(stats["patch_adaptive_major_radius"]),
                float(stats["patch_adaptive_minor_radius"]),
            )
            patch_confidence[finger_index, round_index] = float(stats["patch_confidence"])
            patch_edge_score[finger_index, round_index] = float(stats["patch_edge_score"])
            patch_band_width[finger_index, round_index] = float(stats["patch_band_width"])
            patch_major_extent[finger_index, round_index] = float(stats["patch_major_extent"])
            patch_minor_extent[finger_index, round_index] = float(stats["patch_minor_extent"])
            patch_normal_variance[finger_index, round_index] = float(stats["patch_normal_variance"])
            patch_plane_split_flag[finger_index, round_index] = bool(stats["patch_plane_split_flag"])
            patch_reachability_margin[finger_index, round_index] = float(stats["patch_reachability_margin"])
            patch_target_contact_offset[finger_index, round_index] = float(stats["patch_target_contact_offset"])
            patch_target_contact_offset_ratio[finger_index, round_index] = float(
                stats["patch_target_contact_offset_ratio"]
            )
            patch_adaptive_major_radius[finger_index, round_index] = float(stats["patch_adaptive_major_radius"])
            patch_adaptive_minor_radius[finger_index, round_index] = float(stats["patch_adaptive_minor_radius"])

            patch_points[finger_index, round_index] = subsample_or_repeat(adaptive_points, points_per_patch, rng)
            patch_normals[finger_index, round_index] = subsample_or_repeat(adaptive_normals, points_per_patch, rng)
            patch_point_confidence[finger_index, round_index] = float(stats["patch_confidence"])
            patch_point_band_width[finger_index, round_index] = float(stats["patch_band_width"])

    patch_radii = fill_missing_radii(patch_radii, patch_mask)
    finger_mask = patch_mask.any(axis=1)
    return {
        "patch_points": patch_points.astype(np.float32),
        "patch_point_normals": patch_normals.astype(np.float32),
        "patch_point_confidence": patch_point_confidence.astype(np.float32),
        "patch_point_band_width": patch_point_band_width.astype(np.float32),
        "patch_mask": patch_mask.astype(bool),
        "patch_counts": patch_counts.astype(np.int32),
        "patch_radii": patch_radii.astype(np.float32),
        "patch_centers": centers.astype(np.float32),
        "finger_mask": finger_mask.astype(bool),
        "patch_confidence": patch_confidence.astype(np.float32),
        "patch_edge_score": patch_edge_score.astype(np.float32),
        "patch_band_width": patch_band_width.astype(np.float32),
        "patch_major_extent": patch_major_extent.astype(np.float32),
        "patch_minor_extent": patch_minor_extent.astype(np.float32),
        "patch_normal_variance": patch_normal_variance.astype(np.float32),
        "patch_plane_split_flag": patch_plane_split_flag.astype(bool),
        "patch_reachability_margin": patch_reachability_margin.astype(np.float32),
        "patch_target_contact_offset": patch_target_contact_offset.astype(np.float32),
        "patch_target_contact_offset_ratio": patch_target_contact_offset_ratio.astype(np.float32),
        "patch_adaptive_major_radius": patch_adaptive_major_radius.astype(np.float32),
        "patch_adaptive_minor_radius": patch_adaptive_minor_radius.astype(np.float32),
    }


def build_structured_touch_payload(
    touch_data: dict[str, np.ndarray],
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    points_per_patch: int,
    seed: int,
    adaptive_minor_radius_floor_ratio: float,
    adaptive_minor_radius_scale: float,
    adaptive_major_radius_scale: float,
    adaptive_plane_gap_ratio: float,
    adaptive_min_points_per_patch: int,
    confidence_floor: float,
) -> dict[str, np.ndarray]:
    touch_points_all = np.asarray(touch_data["touch_points"], dtype=np.float32)
    touch_round_ids_all = np.asarray(touch_data["touch_round_ids"], dtype=np.int32)
    touch_finger_ids_all = np.asarray(touch_data["touch_finger_ids"], dtype=np.int32)
    touch_centers_all = np.asarray(touch_data["touch_centers"], dtype=np.float32)
    touch_center_normals_all = np.asarray(touch_data["touch_center_normals"], dtype=np.float32)
    touch_target_points_all = np.asarray(touch_data["touch_target_points"], dtype=np.float32)
    touch_target_normals_all = np.asarray(touch_data["touch_target_normals"], dtype=np.float32)
    touch_contact_points_all = np.asarray(touch_data["touch_contact_points"], dtype=np.float32)
    touch_contact_normals_all = np.asarray(touch_data["touch_contact_normals"], dtype=np.float32)

    surface_tree = cKDTree(np.asarray(surface_points, dtype=np.float32))
    stacked: dict[str, list[np.ndarray]] = {}
    num_views = int(touch_points_all.shape[0])
    for view_index in range(num_views):
        rng = np.random.default_rng(int(seed) + 1009 * (view_index + 1))
        payload = structure_single_view_adaptive(
            touch_points=touch_points_all[view_index],
            round_ids=touch_round_ids_all[view_index],
            finger_ids=touch_finger_ids_all[view_index],
            touch_centers=touch_centers_all[view_index],
            touch_center_normals=touch_center_normals_all[view_index],
            touch_target_points=touch_target_points_all[view_index],
            touch_target_normals=touch_target_normals_all[view_index],
            touch_contact_points=touch_contact_points_all[view_index],
            touch_contact_normals=touch_contact_normals_all[view_index],
            surface_tree=surface_tree,
            surface_normals=np.asarray(surface_normals, dtype=np.float32),
            points_per_patch=points_per_patch,
            rng=rng,
            adaptive_minor_radius_floor_ratio=adaptive_minor_radius_floor_ratio,
            adaptive_minor_radius_scale=adaptive_minor_radius_scale,
            adaptive_major_radius_scale=adaptive_major_radius_scale,
            adaptive_plane_gap_ratio=adaptive_plane_gap_ratio,
            adaptive_min_points_per_patch=adaptive_min_points_per_patch,
            confidence_floor=confidence_floor,
        )
        for key, value in payload.items():
            stacked.setdefault(key, []).append(value)

    result = {key: np.stack(values, axis=0) for key, values in stacked.items()}
    result["touch_view_indices"] = np.arange(num_views, dtype=np.int32)
    result["points_per_patch"] = np.asarray(points_per_patch, dtype=np.int32)
    result["num_touch_views"] = np.asarray(num_views, dtype=np.int32)
    result["adaptive_minor_radius_floor_ratio"] = np.asarray(adaptive_minor_radius_floor_ratio, dtype=np.float32)
    result["adaptive_minor_radius_scale"] = np.asarray(adaptive_minor_radius_scale, dtype=np.float32)
    result["adaptive_major_radius_scale"] = np.asarray(adaptive_major_radius_scale, dtype=np.float32)
    result["adaptive_plane_gap_ratio"] = np.asarray(adaptive_plane_gap_ratio, dtype=np.float32)
    result["adaptive_min_points_per_patch"] = np.asarray(adaptive_min_points_per_patch, dtype=np.int32)
    result["confidence_floor"] = np.asarray(confidence_floor, dtype=np.float32)
    return result
