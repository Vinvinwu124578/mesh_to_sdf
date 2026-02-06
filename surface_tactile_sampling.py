# import os
# import numpy as np
# import trimesh


# # =========================
# # 1. 椭圆指尖 surface patch 采样
# # =========================
# def sample_elliptical_patch_on_surface(
#     mesh,
#     center,
#     normal,
#     area_ratio=0.03,
#     num_points=3000
# ):
#     """
#     在 mesh 表面以 center 为中心，normal 为法向
#     采样一个椭圆形 surface patch
#     """

#     total_area = mesh.area
#     target_area = area_ratio * total_area

#     # 指尖椭圆比例（长:短 = 2:1）
#     a = np.sqrt(target_area / (2 * np.pi))
#     b = a / 2

#     normal = normal / np.linalg.norm(normal)

#     # 构建切平面
#     t1 = np.cross(normal, [1, 0, 0])
#     if np.linalg.norm(t1) < 1e-6:
#         t1 = np.cross(normal, [0, 1, 0])
#     t1 /= np.linalg.norm(t1)
#     t2 = np.cross(normal, t1)

#     # 椭圆内均匀采样
#     u = np.random.rand(num_points)
#     v = np.random.rand(num_points)

#     r = np.sqrt(u)
#     theta = 2 * np.pi * v

#     x = r * np.cos(theta) * a
#     y = r * np.sin(theta) * b

#     offsets = x[:, None] * t1 + y[:, None] * t2
#     query_points = center + offsets

#     # 投影回 mesh 表面
#     surface_points, _, _ = trimesh.proximity.closest_point(mesh, query_points)

#     return surface_points


# # =========================
# # 2. 单个 mesh：10 指 surface tactile 采样
# # =========================
# def sample_tactile_surface_points(
#     mesh,
#     fingers=10,
#     area_ratio_per_finger=0.03,
#     points_per_finger=3000
# ):
#     mesh = mesh.copy()
#     mesh.remove_duplicate_faces()
#     mesh.remove_degenerate_faces()
#     mesh.remove_unreferenced_vertices()

#     # 在表面均匀采样 finger 接触中心
#     centers, face_ids = trimesh.sample.sample_surface(mesh, fingers)
#     normals = mesh.face_normals[face_ids]

#     all_points = []
#     all_sdf = []
#     all_finger_id = []

#     for i in range(fingers):
#         patch = sample_elliptical_patch_on_surface(
#             mesh,
#             center=centers[i],
#             normal=normals[i],
#             area_ratio=area_ratio_per_finger,
#             num_points=points_per_finger
#         )

#         all_points.append(patch)
#         all_sdf.append(np.zeros(len(patch)))        # surface → sdf = 0
#         all_finger_id.append(np.full(len(patch), i))

#     return (
#         np.vstack(all_points),
#         np.concatenate(all_sdf),
#         np.concatenate(all_finger_id)
#     )


# # =========================
# # 3. 单个 OBJ → tactile npz
# # =========================
# def process_single_obj(obj_path, out_path):
#     try:
#         mesh = trimesh.load(obj_path, force="mesh")

#         points, sdf, finger_id = sample_tactile_surface_points(
#             mesh,
#             fingers=10,
#             area_ratio_per_finger=0.03,
#             points_per_finger=3000   # 总点数 ≈ 30k
#         )

#         np.savez(
#             out_path,
#             points=points.astype(np.float32),
#             sdf=sdf.astype(np.float32),
#             finger_id=finger_id.astype(np.int32)
#         )

#         print("Saved:", out_path)

#     except Exception as e:
#         print("Failed:", obj_path, "Error:", e)


# # =========================
# # 4. 批量处理 ModelNet40
# # =========================
# def process_modelnet40(root_dir):
#     for category in os.listdir(root_dir):
#         category_dir = os.path.join(root_dir, category)
#         if not os.path.isdir(category_dir):
#             continue

#         print(f"\nProcessing category: {category}")

#         for split in ["train", "test"]:
#             obj_dir = os.path.join(category_dir, f"{split}_obj")
#             out_dir = os.path.join(category_dir, f"tactile_npz_{split}")

#             if not os.path.isdir(obj_dir):
#                 continue

#             os.makedirs(out_dir, exist_ok=True)

#             for name in os.listdir(obj_dir):
#                 if not name.lower().endswith(".obj"):
#                     continue

#                 obj_path = os.path.join(obj_dir, name)
#                 out_path = os.path.join(
#                     out_dir, name.replace(".obj", ".npz")
#                 )

#                 if os.path.exists(out_path):
#                     print("Skip (exists):", out_path)
#                     continue

#                 process_single_obj(obj_path, out_path)


# # =========================
# # 5. 主入口
# # =========================
# if __name__ == "__main__":
#     root_dir = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40"
#     process_modelnet40(root_dir)
#     print("\nAll tactile surface npz generated.")







# import os
# import numpy as np
# import trimesh


# # =========================
# # 1. 椭圆指尖 surface patch 采样
# # =========================
# def sample_elliptical_patch_on_surface(
#     mesh,
#     center,
#     normal,
#     area_ratio=0.03,
#     num_points=3000
# ):
#     total_area = mesh.area
#     target_area = area_ratio * total_area

#     # 椭圆比例 2:1
#     a = np.sqrt(target_area / (2 * np.pi))
#     b = a / 2

#     normal = normal / np.linalg.norm(normal)

#     # 构建切平面
#     t1 = np.cross(normal, [1, 0, 0])
#     if np.linalg.norm(t1) < 1e-6:
#         t1 = np.cross(normal, [0, 1, 0])
#     t1 /= np.linalg.norm(t1)
#     t2 = np.cross(normal, t1)

#     # 椭圆内采样
#     u = np.random.rand(num_points)
#     v = np.random.rand(num_points)

#     r = np.sqrt(u)
#     theta = 2 * np.pi * v

#     x = r * np.cos(theta) * a
#     y = r * np.sin(theta) * b

#     offsets = x[:, None] * t1 + y[:, None] * t2
#     query_points = center + offsets

#     # 投影回 mesh 表面
#     surface_points, _, _ = trimesh.proximity.closest_point(mesh, query_points)

#     return surface_points


# # =========================
# # 2. 单个 mesh：10 指 tactile 采样
# # =========================
# def sample_tactile_surface_points(
#     mesh,
#     fingers=10,
#     area_ratio_per_finger=0.03,
#     points_per_finger=3000
# ):
#     mesh = mesh.copy()

#     # ✅ 兼容所有 trimesh 版本
#     mesh.process(validate=True)
#     mesh.remove_degenerate_faces()
#     mesh.remove_unreferenced_vertices()

#     centers, face_ids = trimesh.sample.sample_surface(mesh, fingers)
#     normals = mesh.face_normals[face_ids]

#     all_points = []
#     all_sdf = []
#     all_finger_id = []

#     for i in range(fingers):
#         patch = sample_elliptical_patch_on_surface(
#             mesh,
#             center=centers[i],
#             normal=normals[i],
#             area_ratio=area_ratio_per_finger,
#             num_points=points_per_finger
#         )

#         all_points.append(patch)
#         all_sdf.append(np.zeros(len(patch)))
#         all_finger_id.append(np.full(len(patch), i))

#     return (
#         np.vstack(all_points),
#         np.concatenate(all_sdf),
#         np.concatenate(all_finger_id)
#     )


# # =========================
# # 3. 单个 OBJ → tactile npz
# # =========================
# def process_single_obj(obj_path, out_path):
#     try:
#         mesh = trimesh.load(obj_path, force="mesh")

#         points, sdf, finger_id = sample_tactile_surface_points(
#             mesh,
#             fingers=10,
#             area_ratio_per_finger=0.03,
#             points_per_finger=3000
#         )

#         np.savez(
#             out_path,
#             points=points.astype(np.float32),
#             sdf=sdf.astype(np.float32),
#             finger_id=finger_id.astype(np.int32)
#         )

#         print("Saved:", out_path)

#     except Exception as e:
#         print("Failed:", obj_path, "Error:", e)


# # =========================
# # 4. 批量处理 ModelNet40
# # =========================
# def process_modelnet40(root_dir):
#     for category in os.listdir(root_dir):
#         category_dir = os.path.join(root_dir, category)
#         if not os.path.isdir(category_dir):
#             continue

#         print(f"\nProcessing category: {category}")

#         for split in ["train", "test"]:
#             obj_dir = os.path.join(category_dir, f"{split}_obj")
#             out_dir = os.path.join(category_dir, f"tactile_npz_{split}")

#             if not os.path.isdir(obj_dir):
#                 continue

#             os.makedirs(out_dir, exist_ok=True)

#             for name in os.listdir(obj_dir):
#                 if not name.lower().endswith(".obj"):
#                     continue

#                 obj_path = os.path.join(obj_dir, name)
#                 out_path = os.path.join(
#                     out_dir, name.replace(".obj", ".npz")
#                 )

#                 if os.path.exists(out_path):
#                     print("Skip (exists):", out_path)
#                     continue

#                 process_single_obj(obj_path, out_path)


# # =========================
# # 5. 主入口
# # =========================
# if __name__ == "__main__":
#     root_dir = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40"
#     process_modelnet40(root_dir)
#     print("\nAll tactile surface npz generated.")




# import os
# import numpy as np
# import trimesh


# # =========================
# # 1. 在单个三角面上采样椭圆 patch（无 rtree）
# # =========================
# def sample_elliptical_patch_on_face(
#     mesh,
#     face_id,
#     center,
#     area_ratio=0.03,
#     num_points=3000
# ):
#     total_area = mesh.area
#     target_area = area_ratio * total_area

#     # 指尖椭圆比例 2:1
#     a = np.sqrt(target_area / (2 * np.pi))
#     b = a / 2

#     normal = mesh.face_normals[face_id]

#     # 构建切平面
#     t1 = np.cross(normal, [1, 0, 0])
#     if np.linalg.norm(t1) < 1e-6:
#         t1 = np.cross(normal, [0, 1, 0])
#     t1 /= np.linalg.norm(t1)
#     t2 = np.cross(normal, t1)

#     # 椭圆内采样（2D）
#     u = np.random.rand(num_points)
#     v = np.random.rand(num_points)

#     r = np.sqrt(u)
#     theta = 2 * np.pi * v

#     x = r * np.cos(theta) * a
#     y = r * np.sin(theta) * b

#     # 映射到 3D
#     points = center + x[:, None] * t1 + y[:, None] * t2

#     # ⚠️ 关键：不做投影、不用 rtree
#     return points


# # =========================
# # 2. 单个 mesh：10 指 tactile surface 采样
# # =========================
# def sample_tactile_surface_points(
#     mesh,
#     fingers=10,
#     area_ratio_per_finger=0.03,
#     points_per_finger=3000
# ):
#     mesh = mesh.copy()
#     mesh.process(validate=True)

#     centers, face_ids = trimesh.sample.sample_surface(mesh, fingers)

#     all_points = []
#     all_sdf = []
#     all_finger_id = []

#     for i in range(fingers):
#         patch = sample_elliptical_patch_on_face(
#             mesh,
#             face_id=face_ids[i],
#             center=centers[i],
#             area_ratio=area_ratio_per_finger,
#             num_points=points_per_finger
#         )

#         all_points.append(patch)
#         all_sdf.append(np.zeros(len(patch)))        # surface → sdf = 0
#         all_finger_id.append(np.full(len(patch), i))

#     return (
#         np.vstack(all_points),
#         np.concatenate(all_sdf),
#         np.concatenate(all_finger_id)
#     )


# # =========================
# # 3. 单个 OBJ → tactile npz
# # =========================
# def process_single_obj(obj_path, out_path):
#     try:
#         mesh = trimesh.load(obj_path, force="mesh")

#         points, sdf, finger_id = sample_tactile_surface_points(
#             mesh,
#             fingers=10,
#             area_ratio_per_finger=0.03,
#             points_per_finger=3000
#         )

#         np.savez(
#             out_path,
#             points=points.astype(np.float32),
#             sdf=sdf.astype(np.float32),
#             finger_id=finger_id.astype(np.int32)
#         )

#         print("Saved:", out_path)

#     except Exception as e:
#         print("Failed:", obj_path, "Error:", e)


# # =========================
# # 4. 批量处理 ModelNet40
# # =========================
# def process_modelnet40(root_dir):
#     for category in os.listdir(root_dir):
#         category_dir = os.path.join(root_dir, category)
#         if not os.path.isdir(category_dir):
#             continue

#         print(f"\nProcessing category: {category}")

#         for split in ["train", "test"]:
#             obj_dir = os.path.join(category_dir, f"{split}_obj")
#             out_dir = os.path.join(category_dir, f"tactile_npz_{split}")

#             if not os.path.isdir(obj_dir):
#                 continue

#             os.makedirs(out_dir, exist_ok=True)

#             for name in os.listdir(obj_dir):
#                 if not name.lower().endswith(".obj"):
#                     continue

#                 obj_path = os.path.join(obj_dir, name)
#                 out_path = os.path.join(
#                     out_dir, name.replace(".obj", ".npz")
#                 )

#                 if os.path.exists(out_path):
#                     print("Skip (exists):", out_path)
#                     continue

#                 process_single_obj(obj_path, out_path)


# # =========================
# # 5. 主入口
# # =========================
# if __name__ == "__main__":
#     root_dir = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40"
#     process_modelnet40(root_dir)
#     print("\nAll tactile surface npz generated.")


import os
import numpy as np
import trimesh
from collections import deque


# ============================================================
# 1. 从一个起始三角面出发，沿 mesh 表面扩展 patch（核心）
# ============================================================
def collect_surface_patch_faces(mesh, start_face, target_area):
    """
    在 face adjacency 图上扩展，直到累计面积达到 target_area
    返回一组 face indices
    """
    visited = set()
    queue = deque([start_face])
    collected_faces = []
    area_sum = 0.0

    face_areas = mesh.area_faces
    adjacency = mesh.face_adjacency

    # 构建 face -> neighbor faces 映射
    neighbors = {}
    for f1, f2 in adjacency:
        neighbors.setdefault(f1, []).append(f2)
        neighbors.setdefault(f2, []).append(f1)

    while queue and area_sum < target_area:
        f = queue.popleft()
        if f in visited:
            continue

        visited.add(f)
        collected_faces.append(f)
        area_sum += face_areas[f]

        for nb in neighbors.get(f, []):
            if nb not in visited:
                queue.append(nb)

    return collected_faces


# ============================================================
# 2. 在指定的一组三角面上，按面积均匀采样点（严格在表面）
# ============================================================
def sample_points_on_faces(mesh, face_ids, num_points):
    faces = mesh.faces[face_ids]
    vertices = mesh.vertices

    areas = mesh.area_faces[face_ids]
    probs = areas / areas.sum()

    # 按面积比例选 face
    chosen_faces = np.random.choice(
        len(face_ids),
        size=num_points,
        p=probs
    )

    samples = []

    for idx in chosen_faces:
        f = faces[idx]
        v0, v1, v2 = vertices[f]

        # 三角形内均匀采样（重心坐标）
        r1 = np.sqrt(np.random.rand())
        r2 = np.random.rand()

        p = (
            (1 - r1) * v0 +
            r1 * (1 - r2) * v1 +
            r1 * r2 * v2
        )
        samples.append(p)

    return np.asarray(samples)


# ============================================================
# 3. 单个 mesh：10 个“贴合表面”的手指采样
# ============================================================
def sample_tactile_surface_points_on_mesh(
    mesh,
    fingers=10,
    area_ratio_per_finger=0.03,
    points_per_finger=3000
):
    mesh = mesh.copy()
    mesh.process(validate=True)

    total_area = mesh.area

    all_points = []
    all_sdf = []
    all_finger_id = []

    # 为每个 finger 随机选择一个起始面
    start_faces = np.random.choice(
        len(mesh.faces),
        size=fingers,
        replace=False
    )

    for fid, start_face in enumerate(start_faces):
        target_area = area_ratio_per_finger * total_area

        patch_faces = collect_surface_patch_faces(
            mesh,
            start_face,
            target_area
        )

        points = sample_points_on_faces(
            mesh,
            patch_faces,
            points_per_finger
        )

        all_points.append(points)
        all_sdf.append(np.zeros(len(points)))  # surface contact → sdf = 0
        all_finger_id.append(np.full(len(points), fid))

    return (
        np.vstack(all_points),
        np.concatenate(all_sdf),
        np.concatenate(all_finger_id)
    )


# ============================================================
# 4. 单个 OBJ → tactile surface npz
# ============================================================
def process_single_obj(obj_path, out_path):
    try:
        mesh = trimesh.load(obj_path, force="mesh")

        points, sdf, finger_id = sample_tactile_surface_points_on_mesh(
            mesh,
            fingers=10,
            area_ratio_per_finger=0.03,
            points_per_finger=3000   # 总点数 ≈ 30k
        )

        np.savez(
            out_path,
            points=points.astype(np.float32),
            sdf=sdf.astype(np.float32),
            finger_id=finger_id.astype(np.int32)
        )

        print("Saved:", out_path)

    except Exception as e:
        print("Failed:", obj_path, "Error:", e)


# ============================================================
# 5. 批量处理 ModelNet40（train_obj / test_obj）
# ============================================================
def process_modelnet40(root_dir):
    for category in os.listdir(root_dir):
        category_dir = os.path.join(root_dir, category)
        if not os.path.isdir(category_dir):
            continue

        print(f"\nProcessing category: {category}")

        for split in ["train", "test"]:
            obj_dir = os.path.join(category_dir, f"{split}_obj")
            out_dir = os.path.join(category_dir, f"tactile_npz_{split}")

            if not os.path.isdir(obj_dir):
                continue

            os.makedirs(out_dir, exist_ok=True)

            for name in os.listdir(obj_dir):
                if not name.lower().endswith(".obj"):
                    continue

                obj_path = os.path.join(obj_dir, name)
                out_path = os.path.join(
                    out_dir, name.replace(".obj", ".npz")
                )

                if os.path.exists(out_path):
                    print("Skip (exists):", out_path)
                    continue

                process_single_obj(obj_path, out_path)


# ============================================================
# 6. 主入口
# ============================================================
if __name__ == "__main__":
    root_dir = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40"
    process_modelnet40(root_dir)
    print("\nAll tactile surface npz generated.")




# import os
# import numpy as np
# import trimesh
# from collections import deque


# # ============================================================
# # 基础工具：在 mesh face adjacency 上扩展（带排他约束）
# # ============================================================
# def collect_surface_patch_faces_exclusive(
#     mesh,
#     start_face,
#     target_area,
#     forbidden_faces
# ):
#     visited = set()
#     queue = deque([start_face])
#     collected = []
#     area_sum = 0.0

#     face_areas = mesh.area_faces
#     adjacency = mesh.face_adjacency

#     neighbors = {}
#     for f1, f2 in adjacency:
#         neighbors.setdefault(f1, []).append(f2)
#         neighbors.setdefault(f2, []).append(f1)

#     while queue and area_sum < target_area:
#         f = queue.popleft()

#         if f in visited or f in forbidden_faces:
#             continue

#         visited.add(f)
#         collected.append(f)
#         area_sum += face_areas[f]

#         for nb in neighbors.get(f, []):
#             if nb not in visited and nb not in forbidden_faces:
#                 queue.append(nb)

#     return collected


# # ============================================================
# # 在指定 face 集合上，严格在 surface 内采样点
# # ============================================================
# def sample_points_on_faces(mesh, face_ids, num_points):
#     faces = mesh.faces[face_ids]
#     vertices = mesh.vertices

#     areas = mesh.area_faces[face_ids]
#     probs = areas / areas.sum()

#     face_choices = np.random.choice(
#         len(face_ids),
#         size=num_points,
#         p=probs
#     )

#     samples = []

#     for idx in face_choices:
#         f = faces[idx]
#         v0, v1, v2 = vertices[f]

#         r1 = np.sqrt(np.random.rand())
#         r2 = np.random.rand()

#         p = (
#             (1 - r1) * v0 +
#             r1 * (1 - r2) * v1 +
#             r1 * r2 * v2
#         )
#         samples.append(p)

#     return np.asarray(samples)


# # ============================================================
# # 单指：物理一致的触觉接触 patch（不重叠）
# # ============================================================
# def sample_single_finger_patch(
#     mesh,
#     start_face,
#     forbidden_faces,
#     finger_radius,
#     indentation,
#     max_points
# ):
#     face = mesh.faces[start_face]
#     verts = mesh.vertices[face]
#     surface_center = verts.mean(axis=0)
#     normal = mesh.face_normals[start_face]

#     # 手指球心位置
#     finger_center = surface_center + normal * (finger_radius - indentation)

#     # 允许搜索的最大 surface 区域（上限，防止过大）
#     candidate_faces = collect_surface_patch_faces_exclusive(
#         mesh,
#         start_face,
#         target_area=mesh.area * 0.25,
#         forbidden_faces=forbidden_faces
#     )

#     if len(candidate_faces) == 0:
#         return None, []

#     # 先多采样，再筛选
#     candidates = sample_points_on_faces(
#         mesh,
#         candidate_faces,
#         max_points * 4
#     )

#     dists = np.linalg.norm(candidates - finger_center, axis=1)
#     contact_points = candidates[dists <= finger_radius]

#     if len(contact_points) == 0:
#         return None, []

#     if len(contact_points) > max_points:
#         idx = np.random.choice(len(contact_points), max_points, replace=False)
#         contact_points = contact_points[idx]

#     return contact_points, candidate_faces


# # ============================================================
# # 多指：严格不重叠触觉采样
# # ============================================================
# def sample_tactile_surface_points_physical_nonoverlap(
#     mesh,
#     fingers=10,
#     finger_radius=0.05,
#     indentation_range=(0.005, 0.02),
#     max_points_per_finger=3000
# ):
#     mesh = mesh.copy()
#     mesh.process(validate=True)

#     occupied_faces = set()
#     available_faces = set(range(len(mesh.faces)))

#     all_points = []
#     all_sdf = []
#     all_finger_id = []

#     for fid in range(fingers):
#         if not available_faces:
#             break

#         start_face = np.random.choice(list(available_faces))
#         indentation = np.random.uniform(*indentation_range)

#         patch, used_faces = sample_single_finger_patch(
#             mesh,
#             start_face,
#             forbidden_faces=occupied_faces,
#             finger_radius=finger_radius,
#             indentation=indentation,
#             max_points=max_points_per_finger
#         )

#         if patch is None:
#             available_faces.discard(start_face)
#             continue

#         all_points.append(patch)
#         all_sdf.append(np.zeros(len(patch)))
#         all_finger_id.append(np.full(len(patch), fid))

#         for f in used_faces:
#             occupied_faces.add(f)
#             available_faces.discard(f)

#     return (
#         np.vstack(all_points),
#         np.concatenate(all_sdf),
#         np.concatenate(all_finger_id)
#     )


# # ============================================================
# # 单个 OBJ → tactile npz
# # ============================================================
# def process_single_obj(obj_path, out_path):
#     try:
#         mesh = trimesh.load(obj_path, force="mesh")

#         points, sdf, finger_id = sample_tactile_surface_points_physical_nonoverlap(
#             mesh,
#             fingers=10,
#             finger_radius=0.05,
#             indentation_range=(0.005, 0.02),
#             max_points_per_finger=3000
#         )

#         np.savez(
#             out_path,
#             points=points.astype(np.float32),
#             sdf=sdf.astype(np.float32),
#             finger_id=finger_id.astype(np.int32)
#         )

#         print("Saved:", out_path)

#     except Exception as e:
#         print("Failed:", obj_path, "Error:", e)


# # ============================================================
# # 批量处理 ModelNet40
# # ============================================================
# def process_modelnet40(root_dir):
#     for category in os.listdir(root_dir):
#         category_dir = os.path.join(root_dir, category)
#         if not os.path.isdir(category_dir):
#             continue

#         print(f"\nProcessing category: {category}")

#         for split in ["train", "test"]:
#             obj_dir = os.path.join(category_dir, f"{split}_obj")
#             out_dir = os.path.join(category_dir, f"tactile_npz_{split}")

#             if not os.path.isdir(obj_dir):
#                 continue

#             os.makedirs(out_dir, exist_ok=True)

#             for name in os.listdir(obj_dir):
#                 if not name.lower().endswith(".obj"):
#                     continue

#                 obj_path = os.path.join(obj_dir, name)
#                 out_path = os.path.join(
#                     out_dir, name.replace(".obj", ".npz")
#                 )

#                 if os.path.exists(out_path):
#                     continue

#                 process_single_obj(obj_path, out_path)


# # ============================================================
# # 主入口
# # ============================================================
# if __name__ == "__main__":
#     root_dir = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40"
#     process_modelnet40(root_dir)
#     print("\nAll tactile surface npz generated.")
