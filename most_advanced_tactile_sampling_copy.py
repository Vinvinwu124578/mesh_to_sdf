

"""
five_fingers_random_no_constraints_single_object_visualize.py

改动点（按你的要求）：
- 仍然采样 5 个手指
- ✅ 不再“四个近一个远”（不使用手型分布）
- ✅ 不做任何“手指之间”的几何限制（允许中心很近、允许区域重叠）
- 每个手指：独立随机选一个中心点，然后用“半径圆形 + 法向阈值”裁剪 patch，再抽 points_per_finger 个点
- 单物体 + 保存 npz + Open3D 可视化（没有 open3d 就提示安装）

依赖：numpy, trimesh
可视化：pip install open3d
"""

import os
import numpy as np
import trimesh


class SurfacePointCloud:
    def __init__(self, mesh, points, normals=None, scans=None):
        self.mesh = mesh
        self.points = points
        self.normals = normals
        self.scans = scans


def sample_from_mesh(mesh, sample_point_count=800_000, calculate_normals=True, seed=0):
    state = np.random.get_state()
    # np.random.seed(int(seed) & 0xFFFFFFFF)
    try:
        if calculate_normals:
            points, face_indices = mesh.sample(sample_point_count, return_index=True)
            normals = mesh.face_normals[face_indices]
        else:
            points = mesh.sample(sample_point_count, return_index=False)
            normals = None
    finally:
        np.random.set_state(state)
    return SurfacePointCloud(mesh, points=points, normals=normals, scans=None)


def bbox_diag(points):
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    return float(np.linalg.norm(mx - mn))


def extract_finger_patch(points, normals, center_idx, radius, normal_angle_deg=28.0, min_points=150):
    c = points[center_idx]
    nc = normals[center_idx]

    d = np.linalg.norm(points - c[None, :], axis=1)
    spatial_mask = d <= radius

    cos_th = np.cos(np.deg2rad(normal_angle_deg))
    dot = (normals @ nc)
    normal_mask = dot >= cos_th

    mask = spatial_mask & normal_mask
    if int(mask.sum()) < int(min_points):
        return None
    return mask


def tactile_sample_5_random_fingers_no_constraints(
    spc: SurfacePointCloud,
    points_per_finger=3000,
    patch_radius_ratio=0.06,
    normal_angle_deg=28.0,
    min_points=150,
    max_trials_per_finger=200,
    seed=0
):
    """
    5 个手指：每个手指独立随机找一个中心，抽一个圆形 patch。
    手指之间不做任何限制：允许重叠、允许中心很近。
    """
    assert spc.normals is not None, "need normals"

    points = spc.points.astype(np.float32)
    normals = spc.normals.astype(np.float32)
    diag = bbox_diag(points)
    if diag <= 0:
        raise ValueError("degenerate point cloud bbox")

    rng = np.random.default_rng()
    radius = patch_radius_ratio * diag

    all_pts = []
    all_fid = []
    sdf = []

    center_ids = []

    for fid in range(5):
        success = False
        for _ in range(max_trials_per_finger):
            center_idx = int(rng.integers(0, len(points)))
            mask = extract_finger_patch(
                points, normals, center_idx,
                radius=radius,
                normal_angle_deg=normal_angle_deg,
                min_points=min_points
            )
            if mask is None:
                continue

            idx = np.where(mask)[0]
            if len(idx) >= points_per_finger:
                choose = rng.choice(idx, size=points_per_finger, replace=False)
            else:
                choose = rng.choice(idx, size=points_per_finger, replace=True)

            all_pts.append(points[choose])
            all_fid.append(np.full(points_per_finger, fid, dtype=np.int32))
            sdf.append(np.zeros(points_per_finger, dtype=np.float32))
            center_ids.append(center_idx)
            success = True
            break

        if not success:
            raise RuntimeError(
                f"Failed to sample finger {fid}. "
                f"Try increasing patch_radius_ratio (e.g. 0.08~0.12) or sample_point_count."
            )

    sampled_points = np.vstack(all_pts)
    finger_id = np.concatenate(all_fid)
    sdf = np.concatenate(sdf)

    print("[OK] Sampled 5 random fingers with NO inter-finger constraints.",
          f"patch_radius_ratio={patch_radius_ratio}, normal_angle_deg={normal_angle_deg}, min_points={min_points}")

    return sampled_points, sdf, finger_id, points, center_ids


def visualize_open3d(points_all, sampled_points, finger_id, center_ids=None):
    import open3d as o3d

    colors = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
    ], dtype=np.float64)

    # 背景下采样
    step = max(1, len(points_all) // 200000)
    bg_pts = points_all[::step]

    bg = o3d.geometry.PointCloud()
    bg.points = o3d.utility.Vector3dVector(bg_pts.astype(np.float64))
    bg.colors = o3d.utility.Vector3dVector(np.full((len(bg_pts), 3), 0.75, dtype=np.float64))

    fg = o3d.geometry.PointCloud()
    fg.points = o3d.utility.Vector3dVector(sampled_points.astype(np.float64))
    fg.colors = o3d.utility.Vector3dVector(colors[finger_id % 5])

    geoms = [bg, fg]

    # 中心点（黑球）
    if center_ids is not None:
        centers = points_all[np.array(center_ids, dtype=int)]
        for c in centers:
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            s.paint_uniform_color([0, 0, 0])
            s.translate(c.astype(np.float64))
            geoms.append(s)

    print("[INFO] Opening Open3D window (close it to continue)...")
    o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    OBJ_PATH = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/airplane/train_obj/airplane_0001.obj"
    OUT_NPZ = None

    seed = 0

    mesh = trimesh.load(OBJ_PATH, force="mesh")
    mesh.process(validate=True)

    spc = sample_from_mesh(mesh, sample_point_count=800_000, calculate_normals=True, seed=seed)

    sampled_points, sdf, finger_id, points_all, center_ids = tactile_sample_5_random_fingers_no_constraints(
        spc,
        points_per_finger=3000,
        patch_radius_ratio=0.06,   # 看不明显就调大 0.08~0.12
        normal_angle_deg=28.0,
        min_points=150,
        seed=seed
    )

    if OUT_NPZ is not None:
        os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
        np.savez(OUT_NPZ,
                 points=sampled_points.astype(np.float32),
                 sdf=sdf.astype(np.float32),
                 finger_id=finger_id.astype(np.int32))
        print("Saved:", OUT_NPZ)

    try:
        import open3d  # noqa
        visualize_open3d(points_all, sampled_points, finger_id, center_ids=center_ids)
    except Exception as e:
        print("[WARN] Open3D not available. Install with: pip install open3d")
        print("Reason:", e)
