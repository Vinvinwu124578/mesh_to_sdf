# """
# single_object_tactile_sampling_and_visualize.py

# 功能：
# - 只处理一个 OBJ（你指定路径）
# - 从 mesh 表面采样点云 + 法向（基于你给的 sample_from_mesh）
# - 在点云上做 5 指（4近1远）的“圆形指尖区域”采样，并保证指尖可分辨（最小中心间距 + 法向阈值）
# - 保存 npz（可选）
# - 立刻可视化：按 finger_id 上色

# 依赖：numpy, trimesh, matplotlib
# 不依赖 rtree / scipy
# """

# import os
# import numpy as np
# import trimesh
# import matplotlib.pyplot as plt


# # -----------------------------
# # 0) 轻量点云容器
# # -----------------------------
# class SurfacePointCloud:
#     def __init__(self, mesh, points, normals=None, scans=None):
#         self.mesh = mesh
#         self.points = points
#         self.normals = normals
#         self.scans = scans


# # -----------------------------
# # 1) 你的函数：从 mesh 密集采样表面点（可带法向）
# # -----------------------------
# def sample_from_mesh(mesh, sample_point_count=200_000, calculate_normals=True, seed=0):
#     # trimesh.sample() 内部用 np.random，全局种子影响；这里临时设置全局以保证可复现
#     state = np.random.get_state()
#     np.random.seed(int(seed) & 0xFFFFFFFF)

#     try:
#         if calculate_normals:
#             points, face_indices = mesh.sample(sample_point_count, return_index=True)
#             normals = mesh.face_normals[face_indices]
#         else:
#             points = mesh.sample(sample_point_count, return_index=False)
#             normals = None
#     finally:
#         np.random.set_state(state)

#     return SurfacePointCloud(mesh, points=points, normals=normals, scans=None)


# # -----------------------------
# # 2) 几何工具：切平面基
# # -----------------------------
# def tangent_basis(n):
#     n = n / (np.linalg.norm(n) + 1e-12)
#     a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
#     t1 = np.cross(n, a)
#     t1 = t1 / (np.linalg.norm(t1) + 1e-12)
#     t2 = np.cross(n, t1)
#     t2 = t2 / (np.linalg.norm(t2) + 1e-12)
#     return t1, t2


# def bbox_diag(points):
#     mn = points.min(axis=0)
#     mx = points.max(axis=0)
#     return float(np.linalg.norm(mx - mn))


# # -----------------------------
# # 3) 生成“手型”5 个中心：四指近 + 拇指相对远（但整体仍局部）
# # -----------------------------
# def generate_handlike_centers(points, normals, diag, rng,
#                              cluster_radius_ratio=0.10,
#                              thumb_offset_ratio=0.18,
#                              finger_spread_deg=30.0):
#     N = len(points)
#     idx0 = int(rng.integers(0, N))
#     c0 = points[idx0]
#     n0 = normals[idx0]
#     t1, t2 = tangent_basis(n0)

#     cluster_r = cluster_radius_ratio * diag
#     thumb_r = thumb_offset_ratio * diag

#     base = rng.uniform(0, 2*np.pi)
#     spread = np.deg2rad(finger_spread_deg)

#     offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * (spread / 2.0)
#     targets = []
#     for off in offsets:
#         ang = base + off + rng.normal(scale=spread * 0.05)
#         r = rng.uniform(0.6 * cluster_r, 1.0 * cluster_r)
#         targets.append(c0 + r * (np.cos(ang) * t1 + np.sin(ang) * t2))

#     thumb_ang = base + np.pi + rng.normal(scale=0.12)
#     targets.append(c0 + rng.uniform(0.85 * thumb_r, 1.05 * thumb_r) *
#                    (np.cos(thumb_ang) * t1 + np.sin(thumb_ang) * t2))

#     targets = np.asarray(targets, dtype=np.float32)

#     center_indices = []
#     for p in targets:
#         d = np.linalg.norm(points - p[None, :], axis=1)
#         center_indices.append(int(np.argmin(d)))
#     return center_indices


# # -----------------------------
# # 4) 单个指尖 patch：圆形（欧氏半径）+ 法向阈值（避免跨薄片）
# # -----------------------------
# def extract_finger_patch(points, normals, center_idx,
#                          radius, normal_angle_deg=28.0,
#                          min_points=800):
#     c = points[center_idx]
#     nc = normals[center_idx]

#     d = np.linalg.norm(points - c[None, :], axis=1)
#     spatial_mask = d <= radius

#     cos_th = np.cos(np.deg2rad(normal_angle_deg))
#     dot = (normals @ nc)
#     normal_mask = dot >= cos_th

#     mask = spatial_mask & normal_mask
#     if int(mask.sum()) < int(min_points):
#         return None
#     return mask


# # -----------------------------
# # 5) 5 指触觉采样（4近1远 + patch 断开 + “圆形可见”）
# # -----------------------------
# def tactile_sample_handlike(spc: SurfacePointCloud,
#                             points_per_finger=3000,
#                             patch_radius_ratio=0.040,
#                             normal_angle_deg=28.0,
#                             cluster_radius_ratio=0.10,
#                             thumb_offset_ratio=0.18,
#                             finger_spread_deg=30.0,
#                             min_center_sep_ratio=0.070,
#                             max_trials=60,
#                             seed=0):
#     assert spc.normals is not None, "need normals"

#     points = spc.points.astype(np.float32)
#     normals = spc.normals.astype(np.float32)

#     diag = bbox_diag(points)
#     if diag <= 0:
#         raise ValueError("degenerate point cloud bbox")

#     rng = np.random.default_rng(seed)

#     radius = patch_radius_ratio * diag
#     min_center_sep = min_center_sep_ratio * diag

#     for _ in range(max_trials):
#         center_ids = generate_handlike_centers(
#             points, normals, diag, rng,
#             cluster_radius_ratio=cluster_radius_ratio,
#             thumb_offset_ratio=thumb_offset_ratio,
#             finger_spread_deg=finger_spread_deg
#         )

#         centers_xyz = points[center_ids]
#         ok = True
#         for i in range(5):
#             for j in range(i + 1, 5):
#                 if np.linalg.norm(centers_xyz[i] - centers_xyz[j]) < min_center_sep:
#                     ok = False
#                     break
#             if not ok:
#                 break
#         if not ok:
#             continue

#         masks = []
#         for cid in center_ids:
#             m = extract_finger_patch(
#                 points, normals, cid,
#                 radius=radius,
#                 normal_angle_deg=normal_angle_deg,
#                 min_points=max(300, points_per_finger // 6)
#             )
#             if m is None:
#                 break
#             masks.append(m)
#         if len(masks) != 5:
#             continue

#         occupied = np.zeros(len(points), dtype=bool)
#         final_indices = []
#         final_fids = []
#         for fid, m in enumerate(masks):
#             m2 = m & (~occupied)
#             idx = np.where(m2)[0]
#             if len(idx) < max(200, points_per_finger // 8):
#                 break

#             if len(idx) >= points_per_finger:
#                 choose = rng.choice(idx, size=points_per_finger, replace=False)
#             else:
#                 choose = rng.choice(idx, size=points_per_finger, replace=True)

#             final_indices.append(choose)
#             final_fids.append(np.full(points_per_finger, fid, dtype=np.int32))
#             occupied[choose] = True

#         if len(final_indices) != 5:
#             continue

#         all_idx = np.concatenate(final_indices, axis=0)
#         all_fid = np.concatenate(final_fids, axis=0)
#         all_pts = points[all_idx]
#         sdf = np.zeros((len(all_pts),), dtype=np.float32)

#         return all_pts, sdf, all_fid, points, center_ids  # 额外返回原点云与中心，便于可视化

#     raise RuntimeError(
#         "Failed to generate 5 distinct circular finger patches. "
#         "Try increasing patch_radius_ratio or decreasing min_center_sep_ratio."
#     )


# # -----------------------------
# # 6) 可视化：原始点云（灰）+ 采样点（彩）+ 中心点（黑）
# # -----------------------------
# def visualize_open3d_only_tactile(sampled_points, finger_id, centers=None):
#     import open3d as o3d
#     import numpy as np

#     colors = np.array([
#         [1, 0, 0],
#         [0, 1, 0],
#         [0, 0, 1],
#         [1, 1, 0],
#         [1, 0, 1],
#     ], dtype=np.float64)

#     fg = o3d.geometry.PointCloud()
#     fg.points = o3d.utility.Vector3dVector(sampled_points.astype(np.float64))
#     fg.colors = o3d.utility.Vector3dVector(colors[finger_id % 5])

#     geoms = [fg]

#     if centers is not None:
#         for c in centers:
#             s = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
#             s.paint_uniform_color([0, 0, 0])
#             s.translate(c.astype(np.float64))
#             geoms.append(s)

#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     for g in geoms:
#         vis.add_geometry(g)
#     opt = vis.get_render_option()
#     opt.point_size = 6.0  # 关键：把点变大
#     vis.run()
#     vis.destroy_window()



# # -----------------------------
# # 7) 主入口：只跑一个物体
# # -----------------------------
# if __name__ == "__main__":
#     # ✅ 改成你要处理的那个 OBJ
#     OBJ_PATH = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/airplane/train_obj/airplane_0001.obj"

#     # 可选：保存 npz 到哪里（不想保存就设为 None）
#     OUT_NPZ = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/airplane_0001_tactile.npz"

#     seed = 0

#     mesh = trimesh.load(OBJ_PATH, force="mesh")
#     mesh.process(validate=True)

#     # 采样表面点云（建议先 200k 验证可视化，后续想更稳再加大）
#     spc = sample_from_mesh(mesh, sample_point_count=200_000, calculate_normals=True, seed=seed)

#     sampled_points, sdf, finger_id, points_all, center_ids = tactile_sample_handlike(
#         spc,
#         points_per_finger=3000,
#         patch_radius_ratio=0.040,     # 圆形区域大小：看不出圆就调大 0.05~0.07
#         normal_angle_deg=28.0,        # 跨薄片就调小 18~25
#         min_center_sep_ratio=0.070,   # 分不清手指就调大 0.08~0.12
#         seed=seed
#     )

#     # 保存
#     if OUT_NPZ is not None:
#         os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
#         np.savez(OUT_NPZ,
#                  points=sampled_points.astype(np.float32),
#                  sdf=sdf.astype(np.float32),
#                  finger_id=finger_id.astype(np.int32))
#         print("Saved:", OUT_NPZ)

#     # 可视化
#     centers = points_all[np.array(center_ids, dtype=int)]
#     visualize_open3d_only_tactile(sampled_points, finger_id, centers=centers)





"""
single_finger_tactile_sampling_and_visualize.py

改动点：只采样 1 个手指（一个圆形指尖区域），并可视化。
- 基于 trimesh 的 mesh.sample() 得到表面点云 + 法向（你的 sample_from_mesh 思路）
- 在点云上用“欧氏半径 + 法向阈值”裁剪出一个指尖圆形区域
- 不再要求 5 指/不重叠/中心分离，所以不会再因为 airplane 无解而失败
- Open3D 可视化（推荐）；如果你没装 open3d，也提供 matplotlib 备用

依赖：numpy, trimesh
可视化推荐：pip install open3d
"""

import os
import numpy as np
import trimesh


# -----------------------------
# 0) 轻量点云容器
# -----------------------------
class SurfacePointCloud:
    def __init__(self, mesh, points, normals=None, scans=None):
        self.mesh = mesh
        self.points = points
        self.normals = normals
        self.scans = scans


# -----------------------------
# 1) 表面采样（带法向）
# -----------------------------
def sample_from_mesh(mesh, sample_point_count=300_000, calculate_normals=True, seed=0):
    # trimesh.sample 内部用 np.random；这里临时设置全局种子保证可复现
    state = np.random.get_state()
    np.random.seed(int(seed) & 0xFFFFFFFF)
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


# -----------------------------
# 2) 单指尖 patch：圆形（欧氏半径）+ 法向阈值
# -----------------------------
def extract_single_finger_patch(points, normals, center_idx,
                                radius, normal_angle_deg=28.0):
    c = points[center_idx]
    nc = normals[center_idx]

    # 空间圆形
    d = np.linalg.norm(points - c[None, :], axis=1)
    spatial_mask = d <= radius

    # 法向一致性
    cos_th = np.cos(np.deg2rad(normal_angle_deg))
    dot = (normals @ nc)
    normal_mask = dot >= cos_th

    return spatial_mask & normal_mask


# -----------------------------
# 3) 只采样 1 个手指（自动找一个点数足够的中心）
# -----------------------------
def tactile_sample_single_finger(spc: SurfacePointCloud,
                                 points_per_finger=3000,
                                 patch_radius_ratio=0.05,   # airplane 通常要比 0.04 大一点
                                 normal_angle_deg=28.0,
                                 min_points=400,
                                 max_trials=200,
                                 seed=0):
    assert spc.normals is not None, "need normals"

    points = spc.points.astype(np.float32)
    normals = spc.normals.astype(np.float32)
    diag = bbox_diag(points)
    if diag <= 0:
        raise ValueError("degenerate point cloud bbox")

    rng = np.random.default_rng(seed)
    radius = patch_radius_ratio * diag

    # 尝试随机中心，直到 patch 内点数够用
    for _ in range(max_trials):
        center_idx = int(rng.integers(0, len(points)))
        mask = extract_single_finger_patch(points, normals, center_idx,
                                           radius=radius,
                                           normal_angle_deg=normal_angle_deg)
        idx = np.where(mask)[0]
        if len(idx) < min_points:
            continue

        # 从 patch 内选 points_per_finger 个点
        if len(idx) >= points_per_finger:
            choose = rng.choice(idx, size=points_per_finger, replace=False)
        else:
            choose = rng.choice(idx, size=points_per_finger, replace=True)

        sampled_points = points[choose]
        sdf = np.zeros((len(sampled_points),), dtype=np.float32)
        finger_id = np.zeros((len(sampled_points),), dtype=np.int32)  # 全 0（只有一指）

        return sampled_points, sdf, finger_id, points, center_idx

    raise RuntimeError(
        "Failed to sample a single finger patch. "
        "Try increasing patch_radius_ratio (e.g., 0.06~0.10) or increasing sample_point_count."
    )


# -----------------------------
# 4) Open3D 可视化（推荐）
# -----------------------------
def visualize_open3d_single(points_all, sampled_points, center_point=None):
    import open3d as o3d
    import numpy as np

    # 背景点下采样，避免太密
    step = max(1, len(points_all) // 200000)
    bg_pts = points_all[::step]

    bg = o3d.geometry.PointCloud()
    bg.points = o3d.utility.Vector3dVector(bg_pts.astype(np.float64))
    bg.colors = o3d.utility.Vector3dVector(np.full((len(bg_pts), 3), 0.75, dtype=np.float64))

    fg = o3d.geometry.PointCloud()
    fg.points = o3d.utility.Vector3dVector(sampled_points.astype(np.float64))
    fg.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1, 0, 0]], dtype=np.float64), (len(sampled_points), 1)))

    geoms = [bg, fg]

    if center_point is not None:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        s.paint_uniform_color([0, 0, 0])
        s.translate(center_point.astype(np.float64))
        geoms.append(s)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = 6.0
    vis.run()
    vis.destroy_window()


# -----------------------------
# 5) matplotlib 备用可视化（如果你不想装 open3d）
# -----------------------------
def visualize_matplotlib_single(points_all, sampled_points, center_point=None):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    step = max(1, len(points_all) // 150000)
    bg = points_all[::step]
    ax.scatter(bg[:, 0], bg[:, 1], bg[:, 2], c="0.75", s=0.2, alpha=0.25)

    ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2],
               c="r", s=3.0, alpha=0.95)

    if center_point is not None:
        ax.scatter(center_point[0], center_point[1], center_point[2],
                   c="k", s=80)

    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Single-finger tactile patch")
    plt.tight_layout()
    plt.show()


# -----------------------------
# 6) 主入口：只处理一个 OBJ
# -----------------------------
if __name__ == "__main__":
    OBJ_PATH = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/airplane/train_obj/airplane_0001.obj"
    OUT_NPZ = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/airplane_0001_single_finger.npz"  # 可改为 None 不保存

    seed = 0

    mesh = trimesh.load(OBJ_PATH, force="mesh")
    mesh.process(validate=True)

    # airplane 建议点云更密一点
    spc = sample_from_mesh(mesh, sample_point_count=600_000, calculate_normals=True, seed=seed)

    sampled_points, sdf, finger_id, points_all, center_idx = tactile_sample_single_finger(
        spc,
        points_per_finger=3000,
        patch_radius_ratio=0.06,  # 看不见“圆形”就调大到 0.08~0.12
        normal_angle_deg=28.0,    # 跨薄片就调小到 18~25
        min_points=400,
        seed=seed
    )

    # 保存
    if OUT_NPZ is not None:
        os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
        np.savez(OUT_NPZ,
                 points=sampled_points.astype(np.float32),
                 sdf=sdf.astype(np.float32),
                 finger_id=finger_id.astype(np.int32))
        print("Saved:", OUT_NPZ)

    center_point = points_all[int(center_idx)]

    # 可视化：优先 Open3D；没装就用 matplotlib
    try:
        import open3d  # noqa
        visualize_open3d_single(points_all, sampled_points, center_point=center_point)
    except Exception as e:
        print("[WARN] Open3D not available, falling back to matplotlib. Reason:", e)
        visualize_matplotlib_single(points_all, sampled_points, center_point=center_point)
