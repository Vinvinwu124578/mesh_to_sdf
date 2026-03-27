import numpy as np
import trimesh
import open3d as o3d
from scipy.spatial import cKDTree


# =========================================================
# 1. mesh normalization
# =========================================================
def scale_to_unit_sphere(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    max_dist = np.max(distances)

    if max_dist <= 1e-12:
        raise ValueError("Degenerate mesh: max distance is zero.")

    vertices = vertices / max_dist
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)


# =========================================================
# 2. surface point cloud container
# =========================================================
class SurfacePointCloud:
    def __init__(self, mesh, points, normals):
        self.mesh = mesh
        self.points = points
        self.normals = normals


# =========================================================
# 3. sample mesh surface
# =========================================================
def sample_from_mesh(mesh, n=200000):
    points, face_idx = mesh.sample(n, return_index=True)

    normals = mesh.face_normals[face_idx].astype(np.float32)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(norm, 1e-12, None)

    return SurfacePointCloud(
        mesh=mesh,
        points=points.astype(np.float32),
        normals=normals.astype(np.float32),
    )


# =========================================================
# bbox diagonal
# =========================================================
def bbox_diag(points):
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    return float(np.linalg.norm(mx - mn))


# =========================================================
# build raycasting scene
# =========================================================
def build_raycast_scene(mesh):
    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )

    mesh_o3d = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_o3d)
    return scene


# =========================================================
# ray casting (global outer-hit candidates)
# =========================================================
def raycast_outer_hits(mesh, points, num_rays=20000):
    print("[INFO] ray casting")

    scene = build_raycast_scene(mesh)

    rng = np.random.default_rng()

    dirs = rng.normal(size=(num_rays, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    origins = dirs * 3.0
    directions = -dirs

    rays = np.concatenate([origins, directions], axis=1).astype(np.float32)
    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

    ans = scene.cast_rays(rays)
    t_hit = ans["t_hit"].numpy()

    valid = np.isfinite(t_hit)
    hit_points = origins[valid] + directions[valid] * t_hit[valid][:, None]

    print("[INFO] raw ray hits:", len(hit_points))

    tree = cKDTree(points)
    _, ids = tree.query(hit_points, k=1)

    candidate_ids = np.unique(ids.astype(np.int64))

    print("[INFO] unique outer surface candidates:", len(candidate_ids))

    return candidate_ids, hit_points.astype(np.float32)


# =========================================================
# reachable region (global)
# =========================================================
def build_global_reachable_mask(points, hit_points, beam_radius):
    print("[INFO] building reachable region")

    hit_tree = cKDTree(hit_points)
    dist, _ = hit_tree.query(points, k=1, distance_upper_bound=beam_radius)

    reachable_mask = np.isfinite(dist)

    print("[INFO] reachable points:", int(reachable_mask.sum()))
    return reachable_mask


# =========================================================
# tangent patch
# =========================================================
def extract_theoretical_patch(points, normals, center_idx,
                              radius, thickness,
                              normal_angle_deg=28,
                              min_patch_points=150):
    c = points[center_idx]
    n = normals[center_idx]

    v = points - c
    height = v @ n
    v_plane = v - height[:, None] * n
    plane_dist = np.linalg.norm(v_plane, axis=1)

    plane_mask = plane_dist <= radius
    thickness_mask = np.abs(height) <= thickness

    cos_th = np.cos(np.deg2rad(normal_angle_deg))
    normal_mask = (normals @ n) >= cos_th

    mask = plane_mask & thickness_mask & normal_mask

    if mask.sum() < min_patch_points:
        return None

    return mask


# =========================================================
# local visibility filter
# =========================================================
def filter_visible_points(scene,
                          center,
                          center_normal,
                          candidate_points,
                          eps=2e-3,
                          hit_tol=2e-3):
    if len(candidate_points) == 0:
        return np.zeros(0, dtype=bool)

    origin = center + center_normal * eps

    vec = candidate_points - origin[None, :]
    dist = np.linalg.norm(vec, axis=1)

    valid_dir = dist > 1e-12
    visible = np.zeros(len(candidate_points), dtype=bool)

    if not np.any(valid_dir):
        return visible

    dirs = vec[valid_dir] / dist[valid_dir][:, None]
    origins = np.repeat(origin[None, :], len(dirs), axis=0)

    rays = np.concatenate([origins, dirs], axis=1).astype(np.float32)
    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

    ans = scene.cast_rays(rays)
    t_hit = ans["t_hit"].numpy()

    finite_hit = np.isfinite(t_hit)

    valid_indices = np.where(valid_dir)[0]

    ok = np.zeros(len(dirs), dtype=bool)
    ok[finite_hit] = np.abs(t_hit[finite_hit] - dist[valid_dir][finite_hit]) <= hit_tol

    visible[valid_indices] = ok
    return visible


# =========================================================
# patch + reachable + local visibility
# =========================================================
def extract_patch_with_visibility(
        scene,
        points,
        normals,
        reachable_mask,
        center_idx,
        radius,
        thickness,
        normal_angle_deg=28,
        min_patch_points=150,
        min_final_points=80,
        min_reachable_coverage=0.2,
        min_visible_coverage=0.3,
        eps=1e-3,
        hit_tol=3e-3):

    patch_mask = extract_theoretical_patch(
        points,
        normals,
        center_idx,
        radius,
        thickness,
        normal_angle_deg,
        min_patch_points
    )

    if patch_mask is None:
        return None, {
            "reason": "theoretical_patch_too_small",
            "coverage_reachable": 0.0,
            "coverage_visible": 0.0,
            "n_patch": 0,
            "n_reachable": 0,
            "n_visible": 0
        }

    patch_ids = np.where(patch_mask)[0]

    reachable_ids = patch_ids[reachable_mask[patch_ids]]
    coverage_reachable = len(reachable_ids) / max(len(patch_ids), 1)

    if coverage_reachable < min_reachable_coverage:
        return None, {
            "reason": "reachable_coverage_too_low",
            "coverage_reachable": coverage_reachable,
            "coverage_visible": 0.0,
            "n_patch": len(patch_ids),
            "n_reachable": len(reachable_ids),
            "n_visible": 0
        }

    if len(reachable_ids) < min_final_points:
        return None, {
            "reason": "reachable_points_too_few",
            "coverage_reachable": coverage_reachable,
            "coverage_visible": 0.0,
            "n_patch": len(patch_ids),
            "n_reachable": len(reachable_ids),
            "n_visible": 0
        }

    center = points[center_idx]
    center_normal = normals[center_idx]
    candidate_points = points[reachable_ids]

    visible_local = filter_visible_points(
        scene=scene,
        center=center,
        center_normal=center_normal,
        candidate_points=candidate_points,
        eps=eps,
        hit_tol=hit_tol
    )

    visible_ids = reachable_ids[visible_local]
    coverage_visible = len(visible_ids) / max(len(reachable_ids), 1)

    if coverage_visible < min_visible_coverage:
        return None, {
            "reason": "visible_coverage_too_low",
            "coverage_reachable": coverage_reachable,
            "coverage_visible": coverage_visible,
            "n_patch": len(patch_ids),
            "n_reachable": len(reachable_ids),
            "n_visible": len(visible_ids)
        }

    if len(visible_ids) < min_final_points:
        return None, {
            "reason": "visible_points_too_few",
            "coverage_reachable": coverage_reachable,
            "coverage_visible": coverage_visible,
            "n_patch": len(patch_ids),
            "n_reachable": len(reachable_ids),
            "n_visible": len(visible_ids)
        }

    final_mask = np.zeros(len(points), dtype=bool)
    final_mask[visible_ids] = True

    return final_mask, {
        "reason": "ok",
        "coverage_reachable": coverage_reachable,
        "coverage_visible": coverage_visible,
        "n_patch": len(patch_ids),
        "n_reachable": len(reachable_ids),
        "n_visible": len(visible_ids)
    }

# =========================================================
# A. non-overlap helpers
# =========================================================
def compute_patch_overlap_ratio(mask_a, mask_b):
    inter = int(np.count_nonzero(mask_a & mask_b))
    if inter == 0:
        return 0.0

    denom = max(
        min(int(np.count_nonzero(mask_a)), int(np.count_nonzero(mask_b))),
        1
    )
    return float(inter / denom)


def extract_patch_candidates_fixed_center(
    scene,
    points,
    normals,
    reachable_mask,
    center_idx,
    radius,
    thickness,
):
    setting_list = [
        dict(
            normal_angle_deg=28.0,
            min_patch_points=150,
            min_final_points=80,
            min_reachable_coverage=0.20,
            min_visible_coverage=0.30,
        ),
        dict(
            normal_angle_deg=35.0,
            min_patch_points=80,
            min_final_points=30,
            min_reachable_coverage=0.10,
            min_visible_coverage=0.15,
        ),
    ]

    best_ids = np.zeros((0,), dtype=np.int64)
    best_info = {
        "reason": "no_candidate",
        "coverage_reachable": 0.0,
        "coverage_visible": 0.0,
        "n_patch": 0,
        "n_reachable": 0,
        "n_visible": 0,
        "source": "none",
    }

    for setting in setting_list:
        mask, info = extract_patch_with_visibility(
            scene=scene,
            points=points,
            normals=normals,
            reachable_mask=reachable_mask,
            center_idx=center_idx,
            radius=radius,
            thickness=thickness,
            normal_angle_deg=setting["normal_angle_deg"],
            min_patch_points=setting["min_patch_points"],
            min_final_points=setting["min_final_points"],
            min_reachable_coverage=setting["min_reachable_coverage"],
            min_visible_coverage=setting["min_visible_coverage"],
        )

        if mask is not None:
            ids = np.where(mask)[0].astype(np.int64)
            info = dict(info)
            info["source"] = "visible_patch"
            return ids, info

        ids = np.asarray(info["candidate_ids"], dtype=np.int64)
        if len(ids) > len(best_ids):
            best_ids = ids
            best_info = dict(info)
            best_info["source"] = "relaxed_candidates"

    if len(best_ids) > 0:
        return best_ids, best_info

    center_pt = points[center_idx]
    dist = np.linalg.norm(points - center_pt[None, :], axis=1)
    knn_ids = np.argsort(dist)[:128].astype(np.int64)

    fallback_info = dict(best_info)
    fallback_info["source"] = "center_knn_fallback"
    fallback_info["n_visible"] = len(knn_ids)
    return knn_ids, fallback_info


def resolve_multi_finger_candidates(points, center_ids, candidate_ids_per_finger):
    owner = np.full(len(points), -1, dtype=np.int32)
    best_dist = np.full(len(points), np.inf, dtype=np.float32)

    for fid, center_idx in enumerate(center_ids):
        ids = np.asarray(candidate_ids_per_finger[fid], dtype=np.int64)
        if len(ids) == 0:
            continue

        dist = np.linalg.norm(points[ids] - points[center_idx][None, :], axis=1)
        better = dist < best_dist[ids]

        owner[ids[better]] = fid
        best_dist[ids[better]] = dist[better]

    exclusive_ids_per_finger = []
    for fid in range(len(center_ids)):
        ids = np.where(owner == fid)[0].astype(np.int64)
        exclusive_ids_per_finger.append(ids)

    return exclusive_ids_per_finger


# =========================================================
# edge-aware soft contact probability
# =========================================================
def estimate_edge_deformation_factor(points,
                                     normals,
                                     center_idx,
                                     radius,
                                     edge_neighbor_ratio=0.35):
    """
    估计当前中心附近是否接近“高法向变化边缘”。
    返回 edge_factor in [0,1].
    值越大表示越靠近边缘，越容易发生跨边软接触。
    """
    c = points[center_idx]
    n0 = normals[center_idx]

    d = np.linalg.norm(points - c[None, :], axis=1)
    local_mask = d <= (edge_neighbor_ratio * radius)
    local_ids = np.where(local_mask)[0]

    if len(local_ids) < 20:
        return 0.0

    align = normals[local_ids] @ n0
    align = np.clip(align, -1.0, 1.0)

    # 法向变化越大，说明边缘/折角越明显
    normal_variation = 1.0 - np.mean(np.abs(align))

    # 归一到 [0,1]
    edge_factor = np.clip(normal_variation / 0.5, 0.0, 1.0)
    return float(edge_factor)


def compute_soft_contact_probability(points,
                                     normals,
                                     center_idx,
                                     candidate_ids,
                                     radius,
                                     cross_surface_gain=1.2,
                                     edge_neighbor_ratio=0.35):
    """
    这里不是硬阈值，而是估计接触分布概率。

    prob 由三部分构成：
    1) 基础法向重合度
    2) 距离平方衰减
    3) 边缘形变增强：若中心靠近高法向变化边缘，则允许少量跨边表面被采到
    """
    if len(candidate_ids) == 0:
        return np.zeros((0,), dtype=np.float64)

    c = points[center_idx]
    n0 = normals[center_idx]

    pts = points[candidate_ids]
    nrm = normals[candidate_ids]

    v = pts - c[None, :]
    dist = np.linalg.norm(v, axis=1)

    # 距离平方衰减
    radial = 1.0 - dist / max(radius, 1e-8)
    radial = np.clip(radial, 0.0, None) ** 2

    # 法向重合
    align = nrm @ n0
    align_pos = np.clip(align, 0.0, 1.0)

    # 边缘形变程度
    edge_factor = estimate_edge_deformation_factor(
        points=points,
        normals=normals,
        center_idx=center_idx,
        radius=radius,
        edge_neighbor_ratio=edge_neighbor_ratio
    )

    # 对非完全同法向表面，允许有一小部分软接受概率
    # 这里不是取代 align_pos，而是在 edge_factor 大时给一个增强项
    # 这个增强项仍然与法向“部分接近”相关，不允许完全相反法向大概率进入
    soft_cross = np.clip((align + 1.0) * 0.5, 0.0, 1.0)  # [-1,1] -> [0,1]
    soft_cross = soft_cross ** 2

    prob = radial * (
        align_pos + cross_surface_gain * edge_factor * soft_cross
    )

    # 防止概率全 0
    if np.all(prob <= 1e-12):
        prob = np.ones_like(prob, dtype=np.float64)

    prob = prob.astype(np.float64)
    prob /= prob.sum()

    return prob


# =========================================================
# center sampling
# =========================================================
def sample_valid_fixed_centers(scene,
                               points,
                               normals,
                               candidate_ids,
                               reachable_mask,
                               largest_radius,
                               thickness,
                               min_center_dist,
                               num_centers=5,
                               max_trials=5000):
    rng = np.random.default_rng()
    centers = []

    trials = 0
    while len(centers) < num_centers and trials < max_trials:
        trials += 1

        idx = int(rng.choice(candidate_ids))

        ok_dist = True
        for c in centers:
            if np.linalg.norm(points[idx] - points[c]) < min_center_dist:
                ok_dist = False
                break

        if not ok_dist:
            continue

        mask, info = extract_patch_with_visibility(
            scene=scene,
            points=points,
            normals=normals,
            reachable_mask=reachable_mask,
            center_idx=idx,
            radius=largest_radius,
            thickness=thickness
        )

        if mask is None:
            continue

        centers.append(idx)
        print(
            f"[CENTER] accept center {len(centers)-1}: idx={idx}, "
            f"patch={info['n_patch']}, reachable={info['n_reachable']}, visible={info['n_visible']}"
        )

    if len(centers) < num_centers:
        raise RuntimeError(
            f"Only found {len(centers)} valid centers, need {num_centers}. "
            f"Try increasing sample count or relaxing thresholds."
        )

    return np.array(centers, dtype=np.int64)


# =========================================================
# tactile sampling round
# =========================================================
def tactile_sampling_round(spc,
                           scene,
                           reachable_mask,
                           center_ids,
                           radius,
                           thickness,
                           points_per_finger=3000):
    points = spc.points
    normals = spc.normals
    rng = np.random.default_rng()

    all_pts = []
    all_ids = []

    for fid, center_idx in enumerate(center_ids):
        mask, info = extract_patch_with_visibility(
            scene=scene,
            points=points,
            normals=normals,
            reachable_mask=reachable_mask,
            center_idx=center_idx,
            radius=radius,
            thickness=thickness
        )

        if mask is None:
            print(
                f"[Finger {fid}] NO PATCH | reason={info['reason']} | "
                f"patch={info['n_patch']} | reachable={info['n_reachable']} "
                f"({info['coverage_reachable']:.3f}) | "
                f"visible={info['n_visible']} ({info['coverage_visible']:.3f})"
            )
            continue

        idx = np.where(mask)[0]

        prob = compute_soft_contact_probability(
            points=points,
            normals=normals,
            center_idx=center_idx,
            candidate_ids=idx,
            radius=radius,
            cross_surface_gain=0.35,
            edge_neighbor_ratio=0.35
        )

        choose = rng.choice(
            idx,
            points_per_finger,
            replace=len(idx) < points_per_finger,
            p=prob
        )

        all_pts.append(points[choose])
        all_ids.append(np.full(points_per_finger, fid, dtype=np.int32))

        edge_factor = estimate_edge_deformation_factor(
            points=points,
            normals=normals,
            center_idx=center_idx,
            radius=radius,
            edge_neighbor_ratio=0.35
        )

        print(
            f"[Finger {fid}] OK | patch={info['n_patch']} | "
            f"reachable={info['n_reachable']} ({info['coverage_reachable']:.3f}) | "
            f"visible={info['n_visible']} ({info['coverage_visible']:.3f}) | "
            f"edge_factor={edge_factor:.3f} | sampled={points_per_finger}"
        )

    if len(all_pts) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    pts = np.vstack(all_pts).astype(np.float32)
    ids = np.concatenate(all_ids).astype(np.int32)

    return pts, ids


# =========================================================
# visualization
# =========================================================
def visualize(points, finger_points, finger_ids, center_ids):
    geometries = []

    n_bg = min(50000, len(points))
    idx = np.random.choice(len(points), n_bg, replace=False)

    bg = o3d.geometry.PointCloud()
    bg.points = o3d.utility.Vector3dVector(points[idx])
    bg.paint_uniform_color([0.85, 0.85, 0.85])
    geometries.append(bg)

    colors = [
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0.5, 0],
        [1, 0, 1]
    ]

    for fid in range(5):
        pts = finger_points[finger_ids == fid]
        if len(pts) == 0:
            continue

        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(pts)
        p.paint_uniform_color(colors[fid])
        geometries.append(p)

    for cidx in center_ids:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        s.translate(points[cidx])
        s.paint_uniform_color([0, 0, 0])
        geometries.append(s)

    o3d.visualization.draw_geometries(geometries)


# =========================================================
# main
# =========================================================
def process_single_obj(obj_path):
    mesh = trimesh.load(obj_path, force="mesh")
    mesh.process()

    mesh = scale_to_unit_sphere(mesh)
    spc = sample_from_mesh(mesh)

    diag = bbox_diag(spc.points)

    candidate_ids, hit_points = raycast_outer_hits(mesh, spc.points)

    reachable_mask = build_global_reachable_mask(
        spc.points,
        hit_points,
        beam_radius=0.1
    )

    scene = build_raycast_scene(mesh)

    largest_radius = 0.12 * diag
    smallest_radius = 0.03 * diag
    thickness = 0.01 * diag

    center_ids = sample_valid_fixed_centers(
        scene=scene,
        points=spc.points,
        normals=spc.normals,
        candidate_ids=candidate_ids,
        reachable_mask=reachable_mask,
        largest_radius=largest_radius,
        thickness=thickness,
        min_center_dist=2 * smallest_radius
    )

    print("[INFO] fixed center ids:", center_ids.tolist())

    radius_schedule = np.linspace(0.12, 0.03, 5)

    for rid, ratio in enumerate(radius_schedule):
        radius = ratio * diag

        print(f"\n[ROUND {rid}] radius_ratio={ratio:.4f}, radius={radius:.6f}")

        finger_points, finger_ids = tactile_sampling_round(
            spc=spc,
            scene=scene,
            reachable_mask=reachable_mask,
            center_ids=center_ids,
            radius=radius,
            thickness=thickness
        )

        visualize(spc.points, finger_points, finger_ids, center_ids)


# =========================================================
# run
# =========================================================
if __name__ == "__main__":
    OBJ_PATH = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/chair/train_obj/chair_0001.obj"
    process_single_obj(OBJ_PATH)  