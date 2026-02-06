# import os
# import numpy as np
# import trimesh
# from collections import deque


# # ============================================================
# # A. 图工具：face 邻接表
# # ============================================================
# def build_face_neighbors(mesh):
#     neighbors = {}
#     for f1, f2 in mesh.face_adjacency:
#         neighbors.setdefault(int(f1), []).append(int(f2))
#         neighbors.setdefault(int(f2), []).append(int(f1))
#     return neighbors


# # ============================================================
# # B. 在指定 faces 上按面积均匀采样点（严格在表面）
# # ============================================================
# def sample_points_on_faces(mesh, face_ids, num_points, rng=None):
#     if rng is None:
#         rng = np.random.default_rng()

#     face_ids = np.asarray(face_ids, dtype=np.int64)
#     faces = mesh.faces[face_ids]
#     vertices = mesh.vertices

#     areas = mesh.area_faces[face_ids]
#     s = areas.sum()
#     if s <= 0:
#         raise ValueError("Selected faces have zero total area.")
#     probs = areas / s

#     chosen = rng.choice(len(face_ids), size=num_points, p=probs)
#     samples = np.empty((num_points, 3), dtype=np.float64)

#     for i, idx in enumerate(chosen):
#         f = faces[idx]
#         v0, v1, v2 = vertices[f]

#         # 三角形面积均匀采样（重心坐标）
#         r1 = np.sqrt(rng.random())
#         r2 = rng.random()
#         p = (1 - r1) * v0 + r1 * (1 - r2) * v1 + r1 * r2 * v2
#         samples[i] = p

#     return samples


# # ============================================================
# # C. Patch 生长：BFS + 不重叠 + 半径刹车 + 软面积目标
# # ============================================================
# def grow_patch_faces_soft(
#     mesh,
#     start_face,
#     desired_area,
#     neighbors,
#     occupied_faces=None,
#     max_radius=None,
#     area_tolerance=0.50
# ):
#     """
#     从 start_face 沿邻接扩张，形成连通 patch。
#     - desired_area: 软目标（比如 2% 总面积）
#     - area_tolerance: 容忍浮动（0.50 表示 50% 宽松）
#     - max_radius: 限制 patch 不要沿细线跑远（关键）
#     - occupied_faces: 全局占用，保证不同 patch 断开
#     """
#     if occupied_faces is None:
#         occupied_faces = set()

#     centroids = mesh.triangles_center
#     face_areas = mesh.area_faces

#     seed_c = centroids[start_face]
#     queue = deque([int(start_face)])
#     visited = set()
#     collected = []
#     area_sum = 0.0

#     # 软停止阈值：达到 desired_area*(1+tol) 就可以停
#     hard_stop = desired_area * (1.0 + area_tolerance)

#     while queue:
#         f = queue.popleft()
#         if f in visited:
#             continue
#         if f in occupied_faces:
#             continue

#         # 半径刹车：防止穿过窄连接跑远
#         if max_radius is not None:
#             if np.linalg.norm(centroids[f] - seed_c) > max_radius:
#                 continue

#         visited.add(f)
#         collected.append(f)
#         area_sum += float(face_areas[f])

#         # 达到软目标就可以停（不强求严格等于）
#         if area_sum >= desired_area and area_sum <= hard_stop:
#             break
#         # 如果已经超过 hard_stop，也停（防止变太大）
#         if area_sum > hard_stop:
#             break

#         for nb in neighbors.get(f, []):
#             if nb not in visited and nb not in occupied_faces:
#                 queue.append(nb)

#     return collected, area_sum


# # ============================================================
# # D. 手型种子生成：4近1远（拇指更远，但整体仍局部）
# # ============================================================
# def make_tangent_basis(n):
#     """给定法向 n，构造切平面正交基 (t1,t2)"""
#     n = n / (np.linalg.norm(n) + 1e-12)
#     # 选一个不平行的向量
#     a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
#     t1 = np.cross(n, a)
#     t1 = t1 / (np.linalg.norm(t1) + 1e-12)
#     t2 = np.cross(n, t1)
#     t2 = t2 / (np.linalg.norm(t2) + 1e-12)
#     return t1, t2


# def pick_handlike_seed_faces(
#     mesh,
#     rng,
#     diag,
#     cluster_center_face=None,
#     cluster_radius_ratio=0.12,
#     finger_spread_angle_deg=35,
#     finger_sep_ratio=0.05,
#     thumb_radius_ratio=0.22,
#     thumb_angle_offset_deg=180
# ):
#     """
#     返回 5 个 seed faces：
#     - 4 个在 cluster_radius 内（像四指一排）
#     - 1 个拇指在更远的 thumb_radius 位置（但不是全局最远）
#     通过局部切平面上“目标点”→ 投影到 mesh 表面，再取 face_id。
#     """
#     centroids = mesh.triangles_center
#     F = len(centroids)

#     # 选手掌附近的“中心”seed（cluster_center）
#     if cluster_center_face is None:
#         cluster_center_face = int(rng.integers(0, F))
#     c0 = centroids[cluster_center_face]

#     # 用该 face 的法向定义局部切平面
#     n0 = mesh.face_normals[cluster_center_face]
#     t1, t2 = make_tangent_basis(n0)

#     # 参数转长度
#     cluster_r = cluster_radius_ratio * diag
#     thumb_r = thumb_radius_ratio * diag
#     finger_sep = finger_sep_ratio * diag

#     # 在切平面上定义四指方向：围绕某个“前方方向”小范围扇形展开
#     base_ang = rng.uniform(0, 2*np.pi)
#     spread = np.deg2rad(finger_spread_angle_deg)

#     # 四指相对“前方”角度：-1.5, -0.5, 0.5, 1.5 * (spread/2)
#     offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * (spread / 2.0)

#     # 四指的目标点：距离 c0 在 [0.6*cluster_r, cluster_r] 的范围
#     finger_targets = []
#     for off in offsets:
#         ang = base_ang + off + rng.normal(scale=spread * 0.05)
#         r = rng.uniform(0.6 * cluster_r, cluster_r)
#         p = c0 + r * (np.cos(ang) * t1 + np.sin(ang) * t2)
#         finger_targets.append(p)

#     # 拇指：相对前方大致反方向（180°），并且更远一些
#     thumb_ang = base_ang + np.deg2rad(thumb_angle_offset_deg) + rng.normal(scale=spread * 0.08)
#     thumb_r_jit = rng.uniform(0.85 * thumb_r, 1.05 * thumb_r)
#     thumb_target = c0 + thumb_r_jit * (np.cos(thumb_ang) * t1 + np.sin(thumb_ang) * t2)

#     targets = finger_targets + [thumb_target]  # 5 个目标点（在空间中）

#     # 将目标点投影到 mesh 表面，拿 face_id
#     # trimesh.nearest.on_surface 返回 (closest_points, distance, face_id)
#     closest_pts, _, face_ids = mesh.nearest.on_surface(np.asarray(targets))
#     face_ids = [int(f) for f in face_ids]

#     # 去重 + 保证“至少相互分开一点”（避免两个指尖落到同一片）
#     # 简单做法：按顺序保留，若太近则随机扰动目标点重投影
#     seeds = []
#     seed_points = []
#     for i, (fid, cp) in enumerate(zip(face_ids, closest_pts)):
#         if len(seeds) == 0:
#             seeds.append(fid)
#             seed_points.append(cp)
#             continue

#         ok = True
#         for sp in seed_points:
#             if np.linalg.norm(cp - sp) < finger_sep:
#                 ok = False
#                 break
#         if ok and fid not in seeds:
#             seeds.append(fid)
#             seed_points.append(cp)
#         else:
#             # 扰动再投影，最多尝试几次
#             success = False
#             for _ in range(10):
#                 jit = rng.normal(scale=0.15 * cluster_r, size=3)
#                 p_try = targets[i] + jit
#                 cp2, _, fid2 = mesh.nearest.on_surface(p_try[None, :])
#                 cp2 = cp2[0]
#                 fid2 = int(fid2[0])
#                 ok2 = True
#                 for sp in seed_points:
#                     if np.linalg.norm(cp2 - sp) < finger_sep:
#                         ok2 = False
#                         break
#                 if ok2 and fid2 not in seeds:
#                     seeds.append(fid2)
#                     seed_points.append(cp2)
#                     success = True
#                     break
#             if not success:
#                 # 实在不行就先放进去（后面 patch 生长还有 occupied 断开）
#                 if fid not in seeds:
#                     seeds.append(fid)
#                     seed_points.append(cp)

#     # 如果不足 5 个，随机补齐（仍尽量靠近 c0 的局部范围）
#     while len(seeds) < 5:
#         ang = rng.uniform(0, 2*np.pi)
#         r = rng.uniform(0.2*cluster_r, cluster_r)
#         p = c0 + r * (np.cos(ang)*t1 + np.sin(ang)*t2)
#         cp, _, fid = mesh.nearest.on_surface(p[None, :])
#         fid = int(fid[0])
#         if fid not in seeds:
#             seeds.append(fid)

#     return seeds[:5], cluster_center_face


# # ============================================================
# # E. 主函数：按“手型 4近1远”生成 5 个分离 patch，并采样点
# # ============================================================
# def sample_handlike_tactile_surface_points_on_mesh(
#     mesh,
#     fingers=5,
#     area_ratio_per_finger=0.02,     # 你要的：每指尖大概 2%（软目标）
#     points_per_finger=3000,
#     # 控制“整体比较近”的关键：cluster 半径
#     cluster_radius_ratio=0.12,      # 四指聚集尺度（相对 bbox diag）
#     # 拇指相对更远，但仍局部
#     thumb_radius_ratio=0.22,
#     # patch 生长防止沿细线跑远
#     patch_max_radius_ratio=0.10,    # patch 自身半径刹车（相对 diag）
#     area_tolerance=0.70,            # 软面积：允许 +70%/-任意（不足也接受）
#     finger_sep_ratio=0.05,          # 指尖间最小间距（相对 diag）
#     rng=None
# ):
#     if rng is None:
#         rng = np.random.default_rng()

#     mesh = mesh.copy()
#     mesh.process(validate=True)

#     total_area = float(mesh.area)
#     diag = float(np.linalg.norm(mesh.bounding_box.extents))
#     if diag <= 0:
#         raise ValueError("Degenerate mesh bounding box.")

#     neighbors = build_face_neighbors(mesh)

#     desired_area = area_ratio_per_finger * total_area
#     patch_max_radius = patch_max_radius_ratio * diag

#     # 1) 生成符合手型的 5 个种子面（4近1远）
#     seeds, center_face = pick_handlike_seed_faces(
#         mesh,
#         rng=rng,
#         diag=diag,
#         cluster_center_face=None,
#         cluster_radius_ratio=cluster_radius_ratio,
#         finger_spread_angle_deg=35,
#         finger_sep_ratio=finger_sep_ratio,
#         thumb_radius_ratio=thumb_radius_ratio,
#         thumb_angle_offset_deg=180
#     )

#     # 2) 对每个 seed 生长 patch（断开 + 半径刹车 + 软面积）
#     occupied = set()
#     all_points, all_sdf, all_fid = [], [], []

#     for fid, seed_face in enumerate(seeds[:fingers]):
#         patch_faces, got_area = grow_patch_faces_soft(
#             mesh,
#             start_face=seed_face,
#             desired_area=desired_area,
#             neighbors=neighbors,
#             occupied_faces=occupied,
#             max_radius=patch_max_radius,
#             area_tolerance=area_tolerance
#         )

#         if len(patch_faces) == 0:
#             continue

#         # 占用：确保 patch 之间断开（不共享面）
#         occupied.update(patch_faces)

#         # 采样点
#         pts = sample_points_on_faces(mesh, patch_faces, points_per_finger, rng=rng)
#         all_points.append(pts)
#         all_sdf.append(np.zeros(len(pts), dtype=np.float32))
#         all_fid.append(np.full(len(pts), fid, dtype=np.int32))

#     if len(all_points) == 0:
#         raise RuntimeError("No valid patches were generated. Try increasing cluster_radius_ratio or patch_max_radius_ratio.")

#     return (
#         np.vstack(all_points).astype(np.float32),
#         np.concatenate(all_sdf).astype(np.float32),
#         np.concatenate(all_fid).astype(np.int32)
#     )


# # ============================================================
# # F. 单个 OBJ → tactile surface npz
# # ============================================================
# def process_single_obj(obj_path, out_path,
#                        points_per_finger=3000,
#                        seed=0):
#     try:
#         mesh = trimesh.load(obj_path, force="mesh")
#         rng = np.random.default_rng(seed)

#         points, sdf, finger_id = sample_handlike_tactile_surface_points_on_mesh(
#             mesh,
#             fingers=5,
#             area_ratio_per_finger=0.02,     # 每指尖约 2%
#             points_per_finger=points_per_finger,
#             cluster_radius_ratio=0.12,      # 四指聚集程度（越小越“手掌局部”）
#             thumb_radius_ratio=0.22,        # 拇指相对更远一点
#             patch_max_radius_ratio=0.10,    # patch 本身不要扩张太远
#             area_tolerance=0.70,            # 软目标：不强求精确面积
#             finger_sep_ratio=0.05,
#             rng=rng
#         )

#         np.savez(
#             out_path,
#             points=points,
#             sdf=sdf,
#             finger_id=finger_id
#         )

#         print("Saved:", out_path)

#     except Exception as e:
#         print("Failed:", obj_path, "Error:", e)


# # ============================================================
# # G. 批量处理 ModelNet40（train_obj / test_obj）
# # ============================================================
# def process_modelnet40(root_dir, points_per_finger=3000, seed=0):
#     for category in os.listdir(root_dir):
#         category_dir = os.path.join(root_dir, category)
#         if not os.path.isdir(category_dir):
#             continue

#         print(f"\nProcessing category: {category}")

#         for split in ["train", "test"]:
#             obj_dir = os.path.join(category_dir, f"{split}_obj")
#             out_dir = os.path.join(category_dir, f"modified_fingers_tactile_npz_{split}")

#             if not os.path.isdir(obj_dir):
#                 continue

#             os.makedirs(out_dir, exist_ok=True)

#             for name in os.listdir(obj_dir):
#                 if not name.lower().endswith(".obj"):
#                     continue

#                 obj_path = os.path.join(obj_dir, name)
#                 out_path = os.path.join(out_dir, name.replace(".obj", ".npz"))

#                 if os.path.exists(out_path):
#                     print("Skip (exists):", out_path)
#                     continue

#                 obj_seed = (hash((category, split, name)) + seed) & 0xFFFFFFFF
#                 process_single_obj(obj_path, out_path,
#                                    points_per_finger=points_per_finger,
#                                    seed=obj_seed)


# # ============================================================
# # H. 主入口
# # ============================================================
# if __name__ == "__main__":
#     root_dir = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40"
#     process_modelnet40(root_dir, points_per_finger=3000, seed=0)
#     print("\nAll tactile surface npz generated.")



























# import os
# import numpy as np
# import trimesh
# import heapq
# from collections import deque


# # ============================================================
# # 1) 邻接表 + 边权（用 face centroid 距离作为近似测地边权）
# # ============================================================
# def build_face_graph(mesh):
#     neighbors = {}
#     for f1, f2 in mesh.face_adjacency:
#         f1 = int(f1); f2 = int(f2)
#         neighbors.setdefault(f1, []).append(f2)
#         neighbors.setdefault(f2, []).append(f1)
#     return neighbors


# def face_edge_weight(centroids, f1, f2):
#     return float(np.linalg.norm(centroids[f1] - centroids[f2]))


# # ============================================================
# # 2) Dijkstra：计算 seed_face 到所有 face 的近似测地距离
# # ============================================================
# def dijkstra_face_dist(mesh, seed_face, neighbors, max_dist=None):
#     centroids = mesh.triangles_center
#     F = len(centroids)

#     dist = np.full(F, np.inf, dtype=np.float64)
#     dist[seed_face] = 0.0

#     pq = [(0.0, int(seed_face))]  # (distance, face_id)

#     while pq:
#         d, f = heapq.heappop(pq)
#         if d != dist[f]:
#             continue
#         if max_dist is not None and d > max_dist:
#             break

#         for nb in neighbors.get(f, []):
#             w = face_edge_weight(centroids, f, nb)
#             nd = d + w
#             if nd < dist[nb]:
#                 dist[nb] = nd
#                 heapq.heappush(pq, (nd, nb))

#     return dist


# # # ============================================================
# # # 3) “圆形 patch”：取测地距离 <= r 的 faces，并做软面积调半径
# # # ============================================================
# # def collect_geodesic_disk_faces_soft(
# #     mesh,
# #     seed_face,
# #     desired_area,
# #     neighbors,
# #     occupied_faces=None,
# #     r0_ratio=0.06,          # 初始半径（相对 bbox diag）
# #     r_min_ratio=0.02,
# #     r_max_ratio=0.20,
# #     area_tolerance=0.70,    # 允许面积在 [desired*(1-? ), desired*(1+tol)]，不足也可接受
# #     max_iters=8
# # ):
# #     """
# #     返回：(patch_faces, patch_area)
# #     patch_faces 尽量是测地圆盘形状，并且不与 occupied_faces 重叠
# #     """
# #     if occupied_faces is None:
# #         occupied_faces = set()

# #     diag = float(np.linalg.norm(mesh.bounding_box.extents))
# #     if diag <= 0:
# #         raise ValueError("Degenerate mesh.")

# #     face_areas = mesh.area_faces

# #     # 半径搜索范围
# #     r = r0_ratio * diag
# #     r_min = r_min_ratio * diag
# #     r_max = r_max_ratio * diag

# #     # 先跑一次 dijkstra，max_dist 用 r_max 限制，节省时间
# #     dist = dijkstra_face_dist(mesh, seed_face, neighbors, max_dist=r_max)

# #     # 软上界
# #     hard_hi = desired_area * (1.0 + area_tolerance)

# #     best_faces = []
# #     best_area = 0.0

# #     for _ in range(max_iters):
# #         # 选出 dist <= r 的 faces
# #         candidates = np.where(dist <= r)[0].tolist()
# #         # 去掉已占用
# #         faces = [f for f in candidates if f not in occupied_faces]

# #         area = float(face_areas[faces].sum()) if len(faces) else 0.0

# #         # 记录目前最接近 desired_area 的结果（但不超过 hard_hi 太多）
# #         if area > best_area and area <= hard_hi:
# #             best_faces, best_area = faces, area

# #         # 面积调半径（软）
# #         if area < desired_area * 0.8:
# #             r = min(r * 1.35, r_max)
# #         elif area > hard_hi:
# #             r = max(r * 0.75, r_min)
# #         else:
# #             # 在可接受范围就停
# #             best_faces, best_area = faces, area
# #             break

# #     # 兜底：如果仍为空，至少返回 seed_face（且不占用）
# #     if len(best_faces) == 0 and seed_face not in occupied_faces:
# #         best_faces = [int(seed_face)]
# #         best_area = float(face_areas[int(seed_face)])

# #     return best_faces, best_area
# def collect_geodesic_patch_by_fill(
#     mesh,
#     seed_face,
#     desired_area,
#     neighbors,
#     occupied_faces=None,
#     max_dist_ratio=0.20,     # 限制最远扩张距离，防止跨细线跑远
#     area_tolerance=0.70,     # 软目标：允许到 desired*(1+tol) 停
# ):
#     if occupied_faces is None:
#         occupied_faces = set()

#     diag = float(np.linalg.norm(mesh.bounding_box.extents))
#     max_dist = max_dist_ratio * diag

#     face_areas = mesh.area_faces

#     # 1) 先算测地距离（只算到 max_dist 范围内，省时间）
#     dist = dijkstra_face_dist(mesh, seed_face, neighbors, max_dist=max_dist)

#     # 2) 把可达的 face 按 dist 排序
#     valid = np.where(np.isfinite(dist))[0]
#     valid = [int(f) for f in valid if dist[f] <= max_dist and f not in occupied_faces]

#     if len(valid) == 0:
#         if seed_face not in occupied_faces:
#             return [int(seed_face)], float(face_areas[int(seed_face)])
#         return [], 0.0

#     valid.sort(key=lambda f: dist[f])

#     # 3) 从近到远填充到接近 desired_area（软）
#     patch = []
#     area_sum = 0.0
#     hard_hi = desired_area * (1.0 + area_tolerance)

#     for f in valid:
#         a = float(face_areas[f])
#         if a <= 0:
#             continue
#         patch.append(f)
#         area_sum += a

#         # 达到目标就停（不强求精确）
#         if area_sum >= desired_area:
#             break
#         if area_sum > hard_hi:
#             break

#     return patch, area_sum


# # ============================================================
# # 4) 在指定 faces 上按面积均匀采样点（严格在表面）
# # ============================================================
# def sample_points_on_faces(mesh, face_ids, num_points, rng=None):
#     if rng is None:
#         rng = np.random.default_rng()

#     face_ids = np.asarray(face_ids, dtype=np.int64)
#     faces = mesh.faces[face_ids]
#     vertices = mesh.vertices

#     areas = mesh.area_faces[face_ids]
#     s = areas.sum()
#     if s <= 0:
#         raise ValueError("Selected faces have zero total area.")
#     probs = areas / s

#     chosen = rng.choice(len(face_ids), size=num_points, p=probs)
#     samples = np.empty((num_points, 3), dtype=np.float64)

#     for i, idx in enumerate(chosen):
#         f = faces[idx]
#         v0, v1, v2 = vertices[f]
#         r1 = np.sqrt(rng.random())
#         r2 = rng.random()
#         p = (1 - r1) * v0 + r1 * (1 - r2) * v1 + r1 * r2 * v2
#         samples[i] = p

#     return samples.astype(np.float32)


# # ============================================================
# # 5) 手型分布：4近1远（在局部切平面生成目标，然后映射到 face centroid 最近）
# #    不用 rtree：用 face centroid KD 近似（这里直接暴力最近，也够用）
# # ============================================================
# def make_tangent_basis(n):
#     n = n / (np.linalg.norm(n) + 1e-12)
#     a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
#     t1 = np.cross(n, a); t1 /= (np.linalg.norm(t1) + 1e-12)
#     t2 = np.cross(n, t1); t2 /= (np.linalg.norm(t2) + 1e-12)
#     return t1, t2


# def nearest_face_by_centroid(centroids, point):
#     d = np.linalg.norm(centroids - point[None, :], axis=1)
#     return int(np.argmin(d))


# def pick_handlike_seed_faces_centroid(
#     mesh,
#     rng,
#     cluster_radius_ratio=0.12,
#     thumb_radius_ratio=0.22,
#     finger_spread_deg=35,
#     finger_sep_ratio=0.05
# ):
#     centroids = mesh.triangles_center
#     F = len(centroids)
#     diag = float(np.linalg.norm(mesh.bounding_box.extents))

#     # 选一个“手掌中心”face
#     center_face = int(rng.integers(0, F))
#     c0 = centroids[center_face]
#     n0 = mesh.face_normals[center_face]
#     t1, t2 = make_tangent_basis(n0)

#     cluster_r = cluster_radius_ratio * diag
#     thumb_r = thumb_radius_ratio * diag
#     finger_sep = finger_sep_ratio * diag

#     base_ang = rng.uniform(0, 2*np.pi)
#     spread = np.deg2rad(finger_spread_deg)

#     offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * (spread / 2.0)
#     finger_targets = []
#     for off in offsets:
#         ang = base_ang + off
#         r = rng.uniform(0.6 * cluster_r, cluster_r)
#         finger_targets.append(c0 + r * (np.cos(ang)*t1 + np.sin(ang)*t2))

#     thumb_ang = base_ang + np.pi + rng.normal(scale=0.15)
#     thumb_target = c0 + rng.uniform(0.85*thumb_r, 1.05*thumb_r) * (np.cos(thumb_ang)*t1 + np.sin(thumb_ang)*t2)

#     targets = finger_targets + [thumb_target]

#     # 映射到最近 face centroid
#     seeds = []
#     seed_points = []
#     for p in targets:
#         fid = nearest_face_by_centroid(centroids, p)
#         cp = centroids[fid]

#         # 简单去重/最小间距
#         ok = True
#         for sp in seed_points:
#             if np.linalg.norm(cp - sp) < finger_sep:
#                 ok = False
#                 break
#         if ok and fid not in seeds:
#             seeds.append(fid); seed_points.append(cp)
#         else:
#             # 轻微扰动重试
#             for _ in range(10):
#                 p2 = p + rng.normal(scale=0.15*cluster_r, size=3)
#                 fid2 = nearest_face_by_centroid(centroids, p2)
#                 cp2 = centroids[fid2]
#                 ok2 = True
#                 for sp in seed_points:
#                     if np.linalg.norm(cp2 - sp) < finger_sep:
#                         ok2 = False
#                         break
#                 if ok2 and fid2 not in seeds:
#                     seeds.append(fid2); seed_points.append(cp2)
#                     break

#     # 保证 5 个
#     while len(seeds) < 5:
#         fid = int(rng.integers(0, F))
#         if fid not in seeds:
#             seeds.append(fid)

#     return seeds[:5]


# # ============================================================
# # 6) 主采样：5 个“圆形指尖 patch”，4近1远，patch 断开
# # ============================================================
# def sample_tactile_handlike_circular_patches(
#     mesh,
#     points_per_finger=3000,
#     area_ratio_per_finger=0.02,  # 软目标：约 2%
#     rng=None
# ):
#     if rng is None:
#         rng = np.random.default_rng()

#     mesh = mesh.copy()
#     mesh.process(validate=True)

#     total_area = float(mesh.area)
#     desired_area = area_ratio_per_finger * total_area

#     neighbors = build_face_graph(mesh)
#     seeds = pick_handlike_seed_faces_centroid(mesh, rng=rng)

#     occupied = set()
#     all_pts, all_sdf, all_fid = [], [], []

#     for fid, seed_face in enumerate(seeds):
#         patch_faces, patch_area = collect_geodesic_patch_by_fill(
#             mesh,
#             seed_face=seed_face,
#             desired_area=desired_area,
#             neighbors=neighbors,
#             occupied_faces=occupied,
#             # r0_ratio=0.06,
#             # r_min_ratio=0.02,
#             # r_max_ratio=0.20,
#             # area_tolerance=0.70,
#             # max_iters=8
#             max_dist_ratio=0.15,    # ⭐ 控制“圆盘大小 + 不沿细线跑远”
#             area_tolerance=0.70     # ⭐ 软面积，不强求 2%
#         )

#         if len(patch_faces) == 0:
#             continue

#         # 断开：占用这些 faces，别的指尖不能再用
#         occupied.update(patch_faces)

#         pts = sample_points_on_faces(mesh, patch_faces, points_per_finger, rng=rng)
#         all_pts.append(pts)
#         all_sdf.append(np.zeros((len(pts),), dtype=np.float32))
#         all_fid.append(np.full((len(pts),), fid, dtype=np.int32))

#     if len(all_pts) == 0:
#         raise RuntimeError("No patches generated. Try increasing r0_ratio or r_max_ratio.")

#     return (
#         np.vstack(all_pts).astype(np.float32),
#         np.concatenate(all_sdf).astype(np.float32),
#         np.concatenate(all_fid).astype(np.int32)
#     )


# # ============================================================
# # 7) 单个 OBJ → npz
# # ============================================================
# def process_single_obj(obj_path, out_path, points_per_finger=3000, seed=0):
#     try:
#         mesh = trimesh.load(obj_path, force="mesh")
#         rng = np.random.default_rng(seed)

#         points, sdf, finger_id = sample_tactile_handlike_circular_patches(
#             mesh,
#             points_per_finger=points_per_finger,
#             area_ratio_per_finger=0.02,
#             rng=rng
#         )

#         np.savez(out_path, points=points, sdf=sdf, finger_id=finger_id)
#         print("Saved:", out_path)

#     except Exception as e:
#         print("Failed:", obj_path, "Error:", e)


# # ============================================================
# # 8) 批量 ModelNet40
# # ============================================================
# def process_modelnet40(root_dir, points_per_finger=3000, seed=0):
#     for category in os.listdir(root_dir):
#         category_dir = os.path.join(root_dir, category)
#         if not os.path.isdir(category_dir):
#             continue

#         print(f"\nProcessing category: {category}")

#         for split in ["train", "test"]:
#             obj_dir = os.path.join(category_dir, f"{split}_obj")
#             out_dir = os.path.join(category_dir, f"modified_fingers_tactile_npz_{split}")
#             if not os.path.isdir(obj_dir):
#                 continue
#             os.makedirs(out_dir, exist_ok=True)

#             for name in os.listdir(obj_dir):
#                 if not name.lower().endswith(".obj"):
#                     continue

#                 obj_path = os.path.join(obj_dir, name)
#                 out_path = os.path.join(out_dir, name.replace(".obj", ".npz"))

#                 if os.path.exists(out_path):
#                     print("Skip (exists):", out_path)
#                     continue

#                 obj_seed = (hash((category, split, name)) + seed) & 0xFFFFFFFF
#                 process_single_obj(obj_path, out_path, points_per_finger=points_per_finger, seed=obj_seed)


# if __name__ == "__main__":
#     root_dir = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40"
#     process_modelnet40(root_dir, points_per_finger=3000, seed=0)
#     print("\nAll tactile surface npz generated.")



































# import os
# import numpy as np
# import trimesh
# import heapq

# # ============================================================
# # 1) 邻接表构建
# # ============================================================
# def build_face_graph(mesh):
#     neighbors = {}
#     for f1, f2 in mesh.face_adjacency:
#         f1, f2 = int(f1), int(f2)
#         neighbors.setdefault(f1, []).append(f2)
#         neighbors.setdefault(f2, []).append(f1)
#     return neighbors

# def face_edge_weight(centroids, f1, f2):
#     return float(np.linalg.norm(centroids[f1] - centroids[f2]))

# # ============================================================
# # 2) 严格半径限制的 Dijkstra
# # ============================================================
# def dijkstra_face_dist(mesh, seed_face, neighbors, max_dist):
#     centroids = mesh.triangles_center
#     F = len(centroids)
#     dist = np.full(F, np.inf, dtype=np.float64)
#     dist[seed_face] = 0.0
#     pq = [(0.0, int(seed_face))]

#     while pq:
#         d, f = heapq.heappop(pq)
#         if d > dist[f]:
#             continue
#         if d > max_dist:
#             continue

#         for nb in neighbors.get(f, []):
#             w = face_edge_weight(centroids, f, nb)
#             nd = d + w
#             if nd < dist[nb] and nd <= max_dist:
#                 dist[nb] = nd
#                 heapq.heappush(pq, (nd, nb))
#     return dist

# # ============================================================
# # 3) 生成圆形 Patch
# # ============================================================
# def collect_strict_circular_patch(mesh, seed_face, desired_area, neighbors, occupied_faces):
#     # 根据面积计算物理半径 R = sqrt(Area / pi)，并给 15% 的拓扑余量
#     ideal_r = np.sqrt(desired_area / np.pi) * 1.15
    
#     # 限制扩张距离
#     dist = dijkstra_face_dist(mesh, seed_face, neighbors, max_dist=ideal_r)
    
#     valid_faces = np.where(np.isfinite(dist))[0]
#     # 排除已占用
#     candidate_faces = [int(f) for f in valid_faces if f not in occupied_faces]
#     # 按距离排序，确保从中心向外填充
#     candidate_faces.sort(key=lambda f: dist[f])
    
#     patch = []
#     current_area = 0.0
#     # 软上限：允许面积有一定波动以维持圆形
#     max_area_limit = desired_area * 1.2 
    
#     face_areas = mesh.area_faces
#     for f in candidate_faces:
#         a = float(face_areas[f])
#         if current_area + a > max_area_limit:
#             break
#         patch.append(f)
#         current_area += a
        
#     return patch, current_area

# # ============================================================
# # 4) 指尖分布算法（4近1远，大张角）
# # ============================================================
# def make_tangent_basis(n):
#     n = n / (np.linalg.norm(n) + 1e-12)
#     a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
#     t1 = np.cross(n, a); t1 /= (np.linalg.norm(t1) + 1e-12)
#     t2 = np.cross(n, t1); t2 /= (np.linalg.norm(t2) + 1e-12)
#     return t1, t2

# def pick_handlike_seeds(mesh, rng):
#     centroids = mesh.triangles_center
#     diag = float(np.linalg.norm(mesh.bounding_box.extents))
    
#     # 随机选掌心
#     center_face = rng.integers(0, len(centroids))
#     c0 = centroids[center_face]
#     n0 = mesh.face_normals[center_face]
#     t1, t2 = make_tangent_basis(n0)
    
#     # 参数调整：增大间距
#     finger_r = 0.18 * diag   # 四指到中心的距离
#     thumb_r = 0.28 * diag    # 拇指到中心的距离
#     spread = np.deg2rad(80)  # 张角扩大到 80 度
    
#     base_ang = rng.uniform(0, 2*np.pi)
#     offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * (spread / 3.0)
    
#     targets = []
#     for off in offsets:
#         ang = base_ang + off
#         targets.append(c0 + finger_r * (np.cos(ang)*t1 + np.sin(ang)*t2))
    
#     # 拇指在对侧
#     thumb_ang = base_ang + np.pi + rng.normal(scale=0.2)
#     targets.append(c0 + thumb_r * (np.cos(thumb_ang)*t1 + np.sin(thumb_ang)*t2))
    
#     seeds = []
#     for p in targets:
#         dist_sq = np.sum((centroids - p)**2, axis=1)
#         fid = int(np.argmin(dist_sq))
#         seeds.append(fid)
#     return seeds

# # ============================================================
# # 5) 主采样函数
# # ============================================================
# def sample_tactile_circular_patches(mesh, points_per_finger=3000, area_ratio=0.015, rng=None):
#     if rng is None: rng = np.random.default_rng()
    
#     mesh.process(validate=True)
#     neighbors = build_face_graph(mesh)
#     seeds = pick_handlike_seeds(mesh, rng)
    
#     desired_area = (area_ratio * mesh.area)
#     occupied = set()
    
#     all_pts, all_fid = [], []
    
#     for i, seed_face in enumerate(seeds):
#         patch_faces, _ = collect_strict_circular_patch(
#             mesh, seed_face, desired_area, neighbors, occupied
#         )
        
#         if not patch_faces:
#             continue
            
#         occupied.update(patch_faces)
        
#         # 表面均匀采样
#         face_ids = np.array(patch_faces)
#         face_areas = mesh.area_faces[face_ids]
#         p = face_areas / face_areas.sum()
        
#         chosen_faces = rng.choice(face_ids, size=points_per_finger, p=p)
#         tri = mesh.vertices[mesh.faces[chosen_faces]]
        
#         # 重心坐标采样
#         r = np.sqrt(rng.random((points_per_finger, 1)))
#         u = rng.random((points_per_finger, 1))
#         pts = (1 - r) * tri[:,0] + r * (1 - u) * tri[:,1] + r * u * tri[:,2]
        
#         all_pts.append(pts.astype(np.float32))
#         all_fid.append(np.full(len(pts), i, dtype=np.int32))
        
#     if not all_pts:
#         return None
        
#     return np.vstack(all_pts), np.zeros(len(np.vstack(all_pts)), dtype=np.float32), np.concatenate(all_fid)

# # ============================================================
# # 6) 文件处理逻辑
# # ============================================================
# def process_single_obj(obj_path, out_path, seed=0):
#     try:
#         mesh = trimesh.load(obj_path, force="mesh")
#         rng = np.random.default_rng(seed)
        
#         result = sample_tactile_circular_patches(mesh, rng=rng)
#         if result is None: return
        
#         pts, sdf, fid = result
#         np.savez(out_path, points=pts, sdf=sdf, finger_id=fid)
#         print(f"Successfully saved: {os.path.basename(out_path)}")
#     except Exception as e:
#         print(f"Error processing {obj_path}: {e}")

# if __name__ == "__main__":
#     # 请修改为你本地的 ModelNet40 路径
#     root_dir = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40"
    
#     # 测试单个文件，建议先跑一个看看效果
#     # process_single_obj("test.obj", "test.npz")
    
#     # 批量处理逻辑 (保持你原来的结构)
#     for category in os.listdir(root_dir):
#         cat_path = os.path.join(root_dir, category)
#         if not os.path.isdir(cat_path): continue
        
#         for split in ["train", "test"]:
#             obj_dir = os.path.join(cat_path, f"{split}_obj")
#             out_dir = os.path.join(cat_path, f"tactile_circular_npz_{split}")
#             if not os.path.isdir(obj_dir): continue
#             os.makedirs(out_dir, exist_ok=True)
            
#             for name in os.listdir(obj_dir):
#                 if name.endswith(".obj"):
#                     process_single_obj(os.path.join(obj_dir, name), 
#                                      os.path.join(out_dir, name.replace(".obj", ".npz")))


















































import os
import numpy as np
import trimesh

# ============================================================
# 1) 局部坐标系构建
# ============================================================
def make_tangent_basis(n):
    n = n / (np.linalg.norm(n) + 1e-12)
    # 找到一个不与法线平行的向量来做叉乘
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, a); t1 /= (np.linalg.norm(t1) + 1e-12)
    t2 = np.cross(n, t1); t2 /= (np.linalg.norm(t2) + 1e-12)
    return t1, t2

# ============================================================
# 2) 椭圆 Patch 核心算法（投影法）
# ============================================================
def collect_ellipsoid_patch_projection(mesh, seed_face, desired_area, eccentricity=1.5, occupied_faces=None):
    """
    eccentricity: 长短轴比例 (1.0 为正圆, >1.0 为椭圆)
    """
    if occupied_faces is None: occupied_faces = set()
    
    c0 = mesh.triangles_center[seed_face]
    n0 = mesh.face_normals[seed_face]
    t1, t2 = make_tangent_basis(n0)
    
    # 根据面积计算长短轴: Area = pi * a * b, a = b * ecc
    # b = sqrt(Area / (pi * ecc))
    b = np.sqrt(desired_area / (np.pi * eccentricity))
    a = b * eccentricity
    
    # 为每个椭圆随机一个旋转角度，模拟真实指尖
    theta = np.random.uniform(0, 2*np.pi)
    major_axis = np.cos(theta)*t1 + np.sin(theta)*t2
    minor_axis = -np.sin(theta)*t1 + np.cos(theta)*t2

    # 1. 初步筛选：只考虑种子点附近一定范围内的面
    search_radius = a * 1.1
    centroids = mesh.triangles_center
    dists_sq = np.sum((centroids - c0)**2, axis=1)
    potential_indices = np.where(dists_sq < search_radius**2)[0]

    patch = []
    for idx in potential_indices:
        if idx in occupied_faces: continue
        
        # 2. 法线过滤：防止切到模型背面 (夹角 > 60度则跳过)
        if np.dot(mesh.face_normals[idx], n0) < 0.5: continue
        
        # 3. 椭圆投影计算
        v = centroids[idx] - c0
        dist_major = np.dot(v, major_axis)
        dist_minor = np.dot(v, minor_axis)
        
        # 椭圆标准方程
        if (dist_major/a)**2 + (dist_minor/b)**2 <= 1.0:
            patch.append(int(idx))
            
    return patch

# ============================================================
# 3) 手型种子点分布（增大间距与张角）
# ============================================================
def pick_handlike_seeds_fixed(mesh, rng):
    centroids = mesh.triangles_center
    diag = float(np.linalg.norm(mesh.bounding_box.extents))
    
    # 找一个靠近物体表面的中心点
    center_face = rng.integers(0, len(centroids))
    c0, n0 = centroids[center_face], mesh.face_normals[center_face]
    t1, t2 = make_tangent_basis(n0)
    
    # 增大物理跨度，防止重叠
    finger_r = 0.22 * diag 
    thumb_r = 0.32 * diag
    spread = np.deg2rad(90) # 90度张角
    
    base_ang = rng.uniform(0, 2*np.pi)
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * (spread / 3.0)
    
    seeds = []
    for i, off in enumerate(offsets):
        ang = base_ang + off
        target = c0 + finger_r * (np.cos(ang)*t1 + np.sin(ang)*t2)
        seeds.append(int(np.argmin(np.sum((centroids - target)**2, axis=1))))
    
    # 拇指在背面
    t_ang = base_ang + np.pi + rng.normal(0, 0.2)
    t_target = c0 + thumb_r * (np.cos(t_ang)*t1 + np.sin(t_ang)*t2)
    seeds.append(int(np.argmin(np.sum((centroids - t_target)**2, axis=1))))
    
    return seeds

# ============================================================
# 4) 主执行逻辑
# ============================================================
def process_single_obj_ellipsoid(obj_path, out_path, points_per_finger=3000, seed=0):
    try:
        mesh = trimesh.load(obj_path, force="mesh")
        rng = np.random.default_rng(seed)
        
        seeds = pick_handlike_seeds_fixed(mesh, rng)
        desired_area = 0.015 * mesh.area # 每个手指占总面积 1.5%
        
        all_pts, all_fid, occupied = [], [], set()

        for i, seed_face in enumerate(seeds):
            # 这里的 eccentricity=1.8 产生明显的长椭圆效果
            patch_faces = collect_ellipsoid_patch_projection(
                mesh, seed_face, desired_area, eccentricity=1.8, occupied_faces=occupied
            )
            
            if not patch_faces: continue
            occupied.update(patch_faces)

            # 在 patch 内均匀采样点
            face_ids = np.array(patch_faces)
            probs = mesh.area_faces[face_ids] / mesh.area_faces[face_ids].sum()
            chosen = rng.choice(face_ids, size=points_per_finger, p=probs)
            
            # 快速重心采样
            v0, v1, v2 = mesh.vertices[mesh.faces[chosen]].transpose(1, 0, 2)
            r1 = np.sqrt(rng.random((points_per_finger, 1)))
            r2 = rng.random((points_per_finger, 1))
            pts = (1 - r1) * v0 + r1 * (1 - r2) * v1 + r1 * r2 * v2
            
            all_pts.append(pts.astype(np.float32))
            all_fid.append(np.full(len(pts), i, dtype=np.int32))

        if all_pts:
            np.savez(out_path, 
                     points=np.vstack(all_pts), 
                     sdf=np.zeros(len(np.vstack(all_pts)), dtype=np.float32), 
                     finger_id=np.concatenate(all_fid))
            print(f"Done: {os.path.basename(obj_path)}")

    except Exception as e:
        print(f"Failed {obj_path}: {e}")

if __name__ == "__main__":
    # 这里的路径请确保正确
    ROOT = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40"
    
    # 示例运行逻辑
    for category in os.listdir(ROOT):
        cat_dir = os.path.join(ROOT, category)
        if not os.path.isdir(cat_dir): continue
        
        for split in ["train", "test"]:
            obj_in = os.path.join(cat_dir, f"{split}_obj")
            npz_out = os.path.join(cat_dir, f"tactile_ellipsoid_npz_{split}")
            if not os.path.isdir(obj_in): continue
            os.makedirs(npz_out, exist_ok=True)
            
            for f in os.listdir(obj_in):
                if f.endswith(".obj"):
                    process_single_obj_ellipsoid(
                        os.path.join(obj_in, f), 
                        os.path.join(npz_out, f.replace(".obj", ".npz"))
                    )