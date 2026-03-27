# import numpy as np
# import trimesh
# import open3d as o3d
# from scipy.spatial import cKDTree


# # -------------------------------------------------
# # normalize mesh
# # -------------------------------------------------
# def scale_to_unit_sphere(mesh):

#     if isinstance(mesh, trimesh.Scene):
#         mesh = mesh.dump().sum()

#     vertices = mesh.vertices - mesh.bounding_box.centroid
#     distances = np.linalg.norm(vertices, axis=1)
#     vertices /= np.max(distances)

#     return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


# # -------------------------------------------------
# # surface point cloud
# # -------------------------------------------------
# class SurfacePointCloud:

#     def __init__(self, mesh, points, normals):

#         self.mesh = mesh
#         self.points = points
#         self.normals = normals


# # -------------------------------------------------
# # sample mesh surface
# # -------------------------------------------------
# def sample_from_mesh(mesh, n=200000):

#     points, face_idx = mesh.sample(n, return_index=True)
#     normals = mesh.face_normals[face_idx]

#     return SurfacePointCloud(mesh, points, normals)


# # -------------------------------------------------
# # bounding box diag
# # -------------------------------------------------
# def bbox_diag(points):

#     mn = points.min(axis=0)
#     mx = points.max(axis=0)

#     return np.linalg.norm(mx - mn)


# # -------------------------------------------------
# # ray casting + beam radius
# # -------------------------------------------------
# def get_outer_surface_indices(mesh,
#                               points,
#                               num_rays=10000,
#                               beam_radius=0.02):

#     print("[INFO] ray casting")

#     legacy = o3d.geometry.TriangleMesh(
#         o3d.utility.Vector3dVector(mesh.vertices),
#         o3d.utility.Vector3iVector(mesh.faces)
#     )

#     mesh_o3d = o3d.t.geometry.TriangleMesh.from_legacy(legacy)

#     scene = o3d.t.geometry.RaycastingScene()
#     scene.add_triangles(mesh_o3d)

#     dirs = np.random.normal(size=(num_rays,3))
#     dirs /= np.linalg.norm(dirs,axis=1)[:,None]

#     origins = dirs * 3.0
#     directions = -dirs

#     rays = np.concatenate([origins,directions],axis=1)
#     rays = o3d.core.Tensor(rays,dtype=o3d.core.Dtype.Float32)

#     ans = scene.cast_rays(rays)

#     t_hit = ans["t_hit"].numpy()

#     mask = np.isfinite(t_hit)

#     hit_points = origins[mask] + directions[mask]*t_hit[mask][:,None]

#     tree = cKDTree(points)

#     neighbor_ids = tree.query_ball_point(hit_points,r=beam_radius)

#     coverage_mask = np.array([len(i)>10 for i in neighbor_ids])

#     filtered_hits = hit_points[coverage_mask]

#     _, ids = tree.query(filtered_hits,k=1)

#     ids = np.unique(ids)

#     print("[INFO] outer surface candidates:",len(ids))

#     return ids


# # -------------------------------------------------
# # tangent plane patch
# # -------------------------------------------------
# def extract_patch(points,
#                   normals,
#                   center_idx,
#                   radius,
#                   thickness,
#                   normal_angle=28):

#     c = points[center_idx]
#     n = normals[center_idx]

#     v = points - c

#     height = v @ n

#     v_plane = v - height[:,None]*n

#     plane_dist = np.linalg.norm(v_plane,axis=1)

#     plane_mask = plane_dist <= radius
#     thickness_mask = np.abs(height) <= thickness

#     cos_th = np.cos(np.deg2rad(normal_angle))
#     normal_mask = (normals @ n) >= cos_th

#     return plane_mask & thickness_mask & normal_mask


# # -------------------------------------------------
# # tactile sampling
# # -------------------------------------------------
# def tactile_sampling_round(spc,
#                            center_ids,
#                            radius_ratio,
#                            thickness_ratio,
#                            points_per_finger=3000):

#     points = spc.points
#     normals = spc.normals

#     diag = bbox_diag(points)

#     radius = radius_ratio * diag
#     thickness = thickness_ratio * diag

#     rng = np.random.default_rng(0)

#     all_pts = []
#     all_ids = []

#     print("[INFO] radius:",radius)

#     for fid,center_idx in enumerate(center_ids):

#         mask = extract_patch(points,
#                              normals,
#                              center_idx,
#                              radius,
#                              thickness)

#         idx = np.where(mask)[0]

#         choose = rng.choice(idx,
#                             points_per_finger,
#                             replace=len(idx)<points_per_finger)

#         all_pts.append(points[choose])
#         all_ids.append(np.full(points_per_finger,fid))

#     pts = np.vstack(all_pts)
#     ids = np.concatenate(all_ids)

#     return pts,ids


# # -------------------------------------------------
# # visualization
# # -------------------------------------------------
# def visualize(points,finger_points,finger_ids,centers):

#     geometries=[]

#     idx=np.random.choice(len(points),50000,replace=False)

#     bg=o3d.geometry.PointCloud()
#     bg.points=o3d.utility.Vector3dVector(points[idx])
#     bg.paint_uniform_color([0.8,0.8,0.8])

#     geometries.append(bg)

#     colors=[
#         [1,0,0],
#         [0,0,1],
#         [0,1,0],
#         [1,0.5,0],
#         [1,0,1]
#     ]

#     for fid in range(5):

#         pts=finger_points[finger_ids==fid]

#         p=o3d.geometry.PointCloud()
#         p.points=o3d.utility.Vector3dVector(pts)
#         p.paint_uniform_color(colors[fid])

#         geometries.append(p)

#     # center visualization
#     for c in centers:

#         s=o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
#         s.translate(points[c])
#         s.paint_uniform_color([0,0,0])

#         geometries.append(s)

#     o3d.visualization.draw_geometries(geometries)


# # -------------------------------------------------
# # main pipeline
# # -------------------------------------------------
# def process_single_obj(obj_path,
#                        rounds=5,
#                        start_ratio=0.12,
#                        end_ratio=0.03):

#     mesh=trimesh.load(obj_path,force="mesh")
#     mesh.process()

#     mesh=scale_to_unit_sphere(mesh)

#     spc=sample_from_mesh(mesh)

#     outer_ids=get_outer_surface_indices(mesh,spc.points)

#     rng=np.random.default_rng(0)

#     center_ids=rng.choice(outer_ids,5,replace=False)

#     print("[INFO] fixed centers:",center_ids)

#     radius_schedule=np.linspace(start_ratio,end_ratio,rounds)

#     for rid,radius_ratio in enumerate(radius_schedule):

#         print("\nRound",rid)

#         finger_points,finger_ids=tactile_sampling_round(
#             spc,
#             center_ids,
#             radius_ratio,
#             thickness_ratio=0.01
#         )

#         visualize(spc.points,
#                   finger_points,
#                   finger_ids,
#                   center_ids)


# # -------------------------------------------------
# # main
# # -------------------------------------------------
# if __name__=="__main__":

#     OBJ_PATH=r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/sofa/train_obj/sofa_0002.obj"

#     process_single_obj(
#         OBJ_PATH,
#         rounds=5,
#         start_ratio=0.24,
#         end_ratio=0.06
#     )








# import numpy as np
# import trimesh
# import open3d as o3d
# from scipy.spatial import cKDTree


# # -------------------------------------------------
# # normalize mesh
# # -------------------------------------------------
# def scale_to_unit_sphere(mesh):
#     if isinstance(mesh, trimesh.Scene):
#         mesh = mesh.dump().sum()

#     vertices = mesh.vertices - mesh.bounding_box.centroid
#     distances = np.linalg.norm(vertices, axis=1)
#     max_dist = np.max(distances)
#     if max_dist <= 1e-12:
#         raise ValueError("Degenerate mesh: max distance is zero.")

#     vertices = vertices / max_dist
#     return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)


# # -------------------------------------------------
# # surface point cloud container
# # -------------------------------------------------
# class SurfacePointCloud:
#     def __init__(self, mesh, points, normals):
#         self.mesh = mesh
#         self.points = points
#         self.normals = normals


# # -------------------------------------------------
# # sample mesh surface
# # -------------------------------------------------
# def sample_from_mesh(mesh, n=200000, seed=0):
#     state = np.random.get_state()
#     np.random.seed(seed & 0xFFFFFFFF)
#     try:
#         points, face_idx = mesh.sample(n, return_index=True)
#     finally:
#         np.random.set_state(state)

#     normals = mesh.face_normals[face_idx]
#     normals = normals.astype(np.float32)

#     # normalize normals for safety
#     norm = np.linalg.norm(normals, axis=1, keepdims=True)
#     normals = normals / np.clip(norm, 1e-12, None)

#     return SurfacePointCloud(
#         mesh=mesh,
#         points=points.astype(np.float32),
#         normals=normals.astype(np.float32),
#     )


# # -------------------------------------------------
# # bounding box diagonal
# # -------------------------------------------------
# def bbox_diag(points):
#     mn = points.min(axis=0)
#     mx = points.max(axis=0)
#     return float(np.linalg.norm(mx - mn))


# # -------------------------------------------------
# # ray casting + finite beam radius filtering
# # -------------------------------------------------
# def get_outer_surface_indices(
#     mesh,
#     points,
#     num_rays=10000,
#     beam_radius=0.02,
#     min_neighbors=10,
#     seed=0,
# ):
#     print("[INFO] ray casting with beam-radius filtering")

#     legacy = o3d.geometry.TriangleMesh(
#         o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
#         o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
#     )

#     mesh_o3d = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
#     scene = o3d.t.geometry.RaycastingScene()
#     _ = scene.add_triangles(mesh_o3d)

#     rng = np.random.default_rng(seed)

#     dirs = rng.normal(size=(num_rays, 3))
#     dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

#     # mesh is normalized to unit sphere, so radius=3 is safely outside
#     origins = dirs * 3.0
#     directions = -dirs

#     rays = np.concatenate([origins, directions], axis=1).astype(np.float32)
#     rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

#     ans = scene.cast_rays(rays)
#     t_hit = ans["t_hit"].numpy()

#     valid = np.isfinite(t_hit)
#     hit_points = origins[valid] + directions[valid] * t_hit[valid][:, None]

#     print(f"[INFO] raw ray hits: {len(hit_points)}")

#     if len(hit_points) == 0:
#         raise RuntimeError("No ray intersections found.")

#     tree = cKDTree(points)
#     neighbor_ids = tree.query_ball_point(hit_points, r=beam_radius)
#     coverage_mask = np.array([len(ids) >= min_neighbors for ids in neighbor_ids], dtype=bool)

#     filtered_hits = hit_points[coverage_mask]
#     print(f"[INFO] beam-filtered hits: {len(filtered_hits)}")

#     if len(filtered_hits) == 0:
#         raise RuntimeError(
#             "No valid outer-surface hits after beam filtering. "
#             "Try increasing num_rays or beam_radius, or decreasing min_neighbors."
#         )

#     _, ids = tree.query(filtered_hits, k=1)
#     ids = np.unique(ids)

#     print(f"[INFO] outer surface candidate points: {len(ids)}")
#     return ids


# # -------------------------------------------------
# # sample 5 fixed centers with minimum distance
# # guarantee smallest-round patches do not overlap
# # -------------------------------------------------
# def sample_centers_with_min_distance(
#     points,
#     candidate_ids,
#     min_dist,
#     num_centers=5,
#     max_trials=20000,
#     seed=0,
# ):
#     rng = np.random.default_rng(seed)
#     candidate_ids = np.asarray(candidate_ids, dtype=np.int64)

#     if len(candidate_ids) < num_centers:
#         raise RuntimeError("Not enough candidate outer-surface points to sample centers.")

#     chosen = []

#     # random order repeated trials
#     for _ in range(max_trials):
#         idx = int(rng.choice(candidate_ids))
#         p = points[idx]

#         ok = True
#         for c in chosen:
#             if np.linalg.norm(p - points[c]) < min_dist:
#                 ok = False
#                 break

#         if ok:
#             chosen.append(idx)

#         if len(chosen) == num_centers:
#             print(f"[INFO] fixed centers found with min_dist = {min_dist:.6f}")
#             return np.array(chosen, dtype=np.int64)

#     raise RuntimeError(
#         "Failed to find valid finger centers with the requested minimum distance. "
#         "Try reducing end_ratio, reducing the safety factor, increasing num_rays, "
#         "or using a larger object / denser candidate set."
#     )


# # -------------------------------------------------
# # tangent plane circular patch
# # -------------------------------------------------
# def extract_patch(
#     points,
#     normals,
#     center_idx,
#     radius,
#     thickness,
#     normal_angle_deg=28.0,
#     min_points=150,
# ):
#     c = points[center_idx]
#     n = normals[center_idx]
#     n = n / max(np.linalg.norm(n), 1e-12)

#     v = points - c[None, :]

#     # signed distance to tangent plane
#     height = v @ n

#     # projection onto tangent plane
#     v_plane = v - height[:, None] * n[None, :]
#     plane_dist = np.linalg.norm(v_plane, axis=1)

#     plane_mask = plane_dist <= radius
#     thickness_mask = np.abs(height) <= thickness

#     cos_th = np.cos(np.deg2rad(normal_angle_deg))
#     normal_mask = (normals @ n) >= cos_th

#     mask = plane_mask & thickness_mask & normal_mask

#     if int(mask.sum()) < int(min_points):
#         return None

#     return mask


# # -------------------------------------------------
# # tactile sampling for one round
# # fixed centers, shrinking radius across rounds
# # -------------------------------------------------
# def tactile_sampling_round(
#     spc,
#     center_ids,
#     radius_ratio,
#     thickness_ratio,
#     points_per_finger=3000,
#     normal_angle_deg=28.0,
#     min_points=150,
#     seed=0,
# ):
#     points = spc.points
#     normals = spc.normals

#     diag = bbox_diag(points)
#     radius = float(radius_ratio) * diag
#     thickness = float(thickness_ratio) * diag

#     rng = np.random.default_rng(seed)

#     all_pts = []
#     all_ids = []

#     print(f"[INFO] round radius    = {radius:.6f}")
#     print(f"[INFO] round thickness = {thickness:.6f}")

#     for fid, center_idx in enumerate(center_ids):
#         mask = extract_patch(
#             points=points,
#             normals=normals,
#             center_idx=int(center_idx),
#             radius=radius,
#             thickness=thickness,
#             normal_angle_deg=normal_angle_deg,
#             min_points=min_points,
#         )

#         if mask is None:
#             raise RuntimeError(
#                 f"Finger {fid}: patch extraction failed. "
#                 f"Try increasing radius_ratio or thickness_ratio."
#             )

#         idx = np.where(mask)[0]

#         choose = rng.choice(
#             idx,
#             size=points_per_finger,
#             replace=(len(idx) < points_per_finger),
#         )

#         all_pts.append(points[choose])
#         all_ids.append(np.full(points_per_finger, fid, dtype=np.int32))

#         print(
#             f"[INFO] finger {fid}: available={len(idx)}, sampled={points_per_finger}, "
#             f"center_idx={int(center_idx)}"
#         )

#     pts = np.vstack(all_pts).astype(np.float32)
#     ids = np.concatenate(all_ids).astype(np.int32)

#     return pts, ids


# # -------------------------------------------------
# # visualization
# # -------------------------------------------------
# def visualize(points, finger_points, finger_ids, center_ids):
#     geometries = []

#     bg_n = min(50000, len(points))
#     bg_idx = np.random.choice(len(points), bg_n, replace=False)

#     bg = o3d.geometry.PointCloud()
#     bg.points = o3d.utility.Vector3dVector(points[bg_idx].astype(np.float64))
#     bg.paint_uniform_color([0.8, 0.8, 0.8])
#     geometries.append(bg)

#     colors = [
#         [1.0, 0.0, 0.0],
#         [0.0, 0.0, 1.0],
#         [0.0, 1.0, 0.0],
#         [1.0, 0.5, 0.0],
#         [1.0, 0.0, 1.0],
#     ]

#     for fid in range(5):
#         pts = finger_points[finger_ids == fid]
#         p = o3d.geometry.PointCloud()
#         p.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
#         p.paint_uniform_color(colors[fid])
#         geometries.append(p)

#     # show centers as black spheres
#     for cidx in center_ids:
#         s = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
#         s.translate(points[int(cidx)].astype(np.float64))
#         s.paint_uniform_color([0.0, 0.0, 0.0])
#         geometries.append(s)

#     o3d.visualization.draw_geometries(
#         geometries,
#         window_name="Deterministic Tactile Sampling",
#         point_show_normal=False,
#     )


# # -------------------------------------------------
# # main pipeline
# # -------------------------------------------------
# def process_single_obj(
#     obj_path,
#     sample_point_count=200000,
#     num_rays=10000,
#     beam_radius=0.02,
#     min_beam_neighbors=10,
#     rounds=5,
#     start_ratio=0.12,
#     end_ratio=0.03,
#     thickness_ratio=0.01,
#     points_per_finger=3000,
#     normal_angle_deg=28.0,
#     center_seed=0,
#     ray_seed=0,
#     sample_seed=0,
#     final_nonoverlap_safety=2.0,   # >= 2.0 means smallest patches do not overlap
# ):
#     mesh = trimesh.load(obj_path, force="mesh")
#     mesh.process(validate=True)

#     mesh = scale_to_unit_sphere(mesh)

#     spc = sample_from_mesh(mesh, n=sample_point_count, seed=sample_seed)

#     outer_ids = get_outer_surface_indices(
#         mesh=mesh,
#         points=spc.points,
#         num_rays=num_rays,
#         beam_radius=beam_radius,
#         min_neighbors=min_beam_neighbors,
#         seed=ray_seed,
#     )

#     diag = bbox_diag(spc.points)
#     final_radius = float(end_ratio) * diag
#     min_center_dist = float(final_nonoverlap_safety) * final_radius

#     center_ids = sample_centers_with_min_distance(
#         points=spc.points,
#         candidate_ids=outer_ids,
#         min_dist=min_center_dist,
#         num_centers=5,
#         max_trials=20000,
#         seed=center_seed,
#     )

#     print("[INFO] fixed center ids:", center_ids.tolist())

#     radius_schedule = np.linspace(start_ratio, end_ratio, rounds)

#     for rid, radius_ratio in enumerate(radius_schedule):
#         print(f"\n[ROUND {rid}] radius_ratio = {radius_ratio:.6f}")

#         finger_points, finger_ids = tactile_sampling_round(
#             spc=spc,
#             center_ids=center_ids,
#             radius_ratio=float(radius_ratio),
#             thickness_ratio=thickness_ratio,
#             points_per_finger=points_per_finger,
#             normal_angle_deg=normal_angle_deg,
#             min_points=150,
#             seed=rid,  # keep deterministic per round
#         )

#         visualize(
#             points=spc.points,
#             finger_points=finger_points,
#             finger_ids=finger_ids,
#             center_ids=center_ids,
#         )


# # -------------------------------------------------
# # main
# # -------------------------------------------------
# if __name__ == "__main__":
#     OBJ_PATH = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/sofa/train_obj/sofa_0002.obj"

#     process_single_obj(
#         obj_path=OBJ_PATH,
#         sample_point_count=200000,
#         num_rays=10000,
#         beam_radius=0.02,
#         min_beam_neighbors=10,
#         rounds=5,
#         start_ratio=0.12,
#         end_ratio=0.03,
#         thickness_ratio=0.01,
#         points_per_finger=3000,
#         normal_angle_deg=28.0,
#         center_seed=0,
#         ray_seed=0,
#         sample_seed=0,
#         final_nonoverlap_safety=2.0,
#     )







# 加入了unreachable几何限制

# import numpy as np
# import trimesh
# import open3d as o3d
# from scipy.spatial import cKDTree


# # =========================================================
# # 1. mesh normalization
# # =========================================================
# def scale_to_unit_sphere(mesh):
#     if isinstance(mesh, trimesh.Scene):
#         mesh = mesh.dump().sum()

#     vertices = mesh.vertices - mesh.bounding_box.centroid
#     distances = np.linalg.norm(vertices, axis=1)
#     max_dist = np.max(distances)

#     if max_dist <= 1e-12:
#         raise ValueError("Degenerate mesh: max distance is zero.")

#     vertices = vertices / max_dist
#     return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)


# # =========================================================
# # 2. surface point cloud container
# # =========================================================
# class SurfacePointCloud:
#     def __init__(self, mesh, points, normals):
#         self.mesh = mesh
#         self.points = points
#         self.normals = normals


# # =========================================================
# # 3. sample mesh surface
# # =========================================================
# def sample_from_mesh(mesh, n=200000, seed=0):
#     state = np.random.get_state()
#     np.random.seed(seed & 0xFFFFFFFF)
#     try:
#         points, face_idx = mesh.sample(n, return_index=True)
#     finally:
#         np.random.set_state(state)

#     normals = mesh.face_normals[face_idx].astype(np.float32)
#     norm = np.linalg.norm(normals, axis=1, keepdims=True)
#     normals = normals / np.clip(norm, 1e-12, None)

#     return SurfacePointCloud(
#         mesh=mesh,
#         points=points.astype(np.float32),
#         normals=normals.astype(np.float32),
#     )


# # =========================================================
# # 4. bbox diagonal
# # =========================================================
# def bbox_diag(points):
#     mn = points.min(axis=0)
#     mx = points.max(axis=0)
#     return float(np.linalg.norm(mx - mn))


# # =========================================================
# # 5. Open3D ray casting
# #    returns:
# #      - unique candidate center ids
# #      - hit points (first-hit surface points)
# # =========================================================
# def raycast_outer_hits(
#     mesh,
#     points,
#     num_rays=20000,
#     seed=0,
# ):
#     print("[INFO] ray casting")

#     legacy = o3d.geometry.TriangleMesh(
#         o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
#         o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
#     )

#     mesh_o3d = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
#     scene = o3d.t.geometry.RaycastingScene()
#     _ = scene.add_triangles(mesh_o3d)

#     rng = np.random.default_rng(seed)

#     dirs = rng.normal(size=(num_rays, 3))
#     dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

#     # normalized mesh is inside unit sphere, so origins at radius=3 are outside
#     origins = dirs * 3.0
#     directions = -dirs

#     rays = np.concatenate([origins, directions], axis=1).astype(np.float32)
#     rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

#     ans = scene.cast_rays(rays)
#     t_hit = ans["t_hit"].numpy()

#     valid = np.isfinite(t_hit)
#     hit_points = origins[valid] + directions[valid] * t_hit[valid][:, None]

#     print(f"[INFO] raw ray hits: {len(hit_points)}")
#     if len(hit_points) == 0:
#         raise RuntimeError("No ray intersections found.")

#     # map hit points to sampled surface points
#     point_tree = cKDTree(points)
#     _, nearest_ids = point_tree.query(hit_points, k=1)
#     candidate_ids = np.unique(nearest_ids.astype(np.int64))

#     print(f"[INFO] unique outer-surface candidates: {len(candidate_ids)}")
#     return candidate_ids, hit_points.astype(np.float32)


# # =========================================================
# # 6. build global reachable mask from UNION OF ALL BEAMS
# #
# #    Important:
# #    user requirement = not a single beam, but the union of all beams
# #
# #    We approximate the reachable surface region as:
# #      any sampled point within beam_radius of ANY first-hit point
# #
# #    That gives:
# #      reachable_region = union of all beam footprints on surface
# # =========================================================
# def build_global_reachable_mask(points, hit_points, beam_radius):
#     print("[INFO] building global reachable region from all beam hits")

#     hit_tree = cKDTree(hit_points)
#     dist, _ = hit_tree.query(points, k=1, distance_upper_bound=beam_radius)

#     reachable_mask = np.isfinite(dist)
#     reachable_count = int(reachable_mask.sum())

#     print(f"[INFO] reachable surface points: {reachable_count} / {len(points)}")
#     return reachable_mask


# # =========================================================
# # 7. tangent plane circular patch
# #    returns theoretical patch mask only
# # =========================================================
# def extract_theoretical_patch(
#     points,
#     normals,
#     center_idx,
#     radius,
#     thickness,
#     normal_angle_deg=28.0,
#     min_patch_points=150,
# ):
#     c = points[center_idx]
#     n = normals[center_idx]
#     n = n / max(np.linalg.norm(n), 1e-12)

#     v = points - c[None, :]

#     # signed distance to tangent plane
#     height = v @ n

#     # projection to tangent plane
#     v_plane = v - height[:, None] * n[None, :]
#     plane_dist = np.linalg.norm(v_plane, axis=1)

#     plane_mask = plane_dist <= radius
#     thickness_mask = np.abs(height) <= thickness

#     cos_th = np.cos(np.deg2rad(normal_angle_deg))
#     normal_mask = (normals @ n) >= cos_th

#     patch_mask = plane_mask & thickness_mask & normal_mask

#     if int(patch_mask.sum()) < int(min_patch_points):
#         return None

#     return patch_mask


# # =========================================================
# # 8. theoretical patch + global reachable coverage constraint
# #
# #    rule:
# #      - build theoretical tangent-plane patch
# #      - coverage = reachable points inside patch / all patch points
# #      - only valid if coverage >= min_reachable_coverage
# #      - final returned mask keeps ONLY reachable points
# # =========================================================
# def extract_patch_with_global_reachability(
#     points,
#     normals,
#     reachable_mask,
#     center_idx,
#     radius,
#     thickness,
#     normal_angle_deg=28.0,
#     min_patch_points=150,
#     min_final_points=80,
#     min_reachable_coverage=0.5,
# ):
#     patch_mask = extract_theoretical_patch(
#         points=points,
#         normals=normals,
#         center_idx=center_idx,
#         radius=radius,
#         thickness=thickness,
#         normal_angle_deg=normal_angle_deg,
#         min_patch_points=min_patch_points,
#     )

#     if patch_mask is None:
#         return None, 0.0, 0, 0

#     patch_ids = np.where(patch_mask)[0]
#     reachable_ids = patch_ids[reachable_mask[patch_ids]]

#     coverage = float(len(reachable_ids)) / float(len(patch_ids))

#     if coverage < min_reachable_coverage:
#         return None, coverage, len(patch_ids), len(reachable_ids)

#     if len(reachable_ids) < min_final_points:
#         return None, coverage, len(patch_ids), len(reachable_ids)

#     final_mask = np.zeros(len(points), dtype=bool)
#     final_mask[reachable_ids] = True

#     return final_mask, coverage, len(patch_ids), len(reachable_ids)


# # =========================================================
# # 9. choose 5 fixed centers
# #    constraints:
# #      - center must come from candidate_ids
# #      - center must satisfy coverage constraint at the LARGEST radius
# #      - center distance must be large enough so final round does not overlap
# # =========================================================
# def sample_valid_fixed_centers(
#     points,
#     normals,
#     candidate_ids,
#     reachable_mask,
#     largest_radius,
#     thickness,
#     min_center_dist,
#     num_centers=5,
#     normal_angle_deg=28.0,
#     min_patch_points=150,
#     min_final_points=80,
#     min_reachable_coverage=0.5,
#     seed=0,
#     max_trials=50000,
# ):
#     rng = np.random.default_rng(seed)
#     candidate_ids = np.asarray(candidate_ids, dtype=np.int64)

#     if len(candidate_ids) < num_centers:
#         raise RuntimeError("Not enough candidate points to sample centers.")

#     chosen = []
#     chosen_set = set()

#     order = rng.permutation(candidate_ids)

#     trials = 0
#     ptr = 0

#     while trials < max_trials:
#         if ptr >= len(order):
#             order = rng.permutation(candidate_ids)
#             ptr = 0

#         idx = int(order[ptr])
#         ptr += 1
#         trials += 1

#         if idx in chosen_set:
#             continue

#         p = points[idx]

#         # distance constraint for final-round non-overlap
#         ok = True
#         for c in chosen:
#             if np.linalg.norm(p - points[c]) < min_center_dist:
#                 ok = False
#                 break

#         if not ok:
#             continue

#         # largest-round coverage check
#         mask, coverage, n_patch, n_reach = extract_patch_with_global_reachability(
#             points=points,
#             normals=normals,
#             reachable_mask=reachable_mask,
#             center_idx=idx,
#             radius=largest_radius,
#             thickness=thickness,
#             normal_angle_deg=normal_angle_deg,
#             min_patch_points=min_patch_points,
#             min_final_points=min_final_points,
#             min_reachable_coverage=min_reachable_coverage,
#         )

#         if mask is None:
#             continue

#         chosen.append(idx)
#         chosen_set.add(idx)

#         print(
#             f"[INFO] accepted center {len(chosen)-1}: idx={idx}, "
#             f"largest-round coverage={coverage:.3f}, "
#             f"patch={n_patch}, reachable_patch={n_reach}"
#         )

#         if len(chosen) == num_centers:
#             print(f"[INFO] fixed centers found with min_dist = {min_center_dist:.6f}")
#             return np.array(chosen, dtype=np.int64)

#     raise RuntimeError(
#         "Failed to find 5 valid fixed centers. "
#         "Try increasing num_rays, increasing beam_radius, reducing start_ratio, "
#         "reducing min_reachable_coverage, or reducing min_center_dist."
#     )


# # =========================================================
# # 10. one round tactile sampling
# #     fixed centers, shrinking radius across rounds
# #     sampling only from reachable part of patch
# # =========================================================
# def tactile_sampling_round(
#     spc,
#     reachable_mask,
#     center_ids,
#     radius,
#     thickness,
#     points_per_finger=3000,
#     normal_angle_deg=28.0,
#     min_patch_points=150,
#     min_final_points=80,
#     min_reachable_coverage=0.5,
#     seed=0,
# ):
#     points = spc.points
#     normals = spc.normals
#     rng = np.random.default_rng(seed)

#     all_pts = []
#     all_ids = []

#     print(f"[INFO] round radius       = {radius:.6f}")
#     print(f"[INFO] round thickness    = {thickness:.6f}")

#     for fid, center_idx in enumerate(center_ids):
#         mask, coverage, n_patch, n_reach = extract_patch_with_global_reachability(
#             points=points,
#             normals=normals,
#             reachable_mask=reachable_mask,
#             center_idx=int(center_idx),
#             radius=radius,
#             thickness=thickness,
#             normal_angle_deg=normal_angle_deg,
#             min_patch_points=min_patch_points,
#             min_final_points=min_final_points,
#             min_reachable_coverage=min_reachable_coverage,
#         )

#         if mask is None:
#             raise RuntimeError(
#                 f"Finger {fid}: patch invalid under global reachable-region constraint. "
#                 f"Try reducing radius, increasing beam_radius, or choosing different centers."
#             )

#         idx = np.where(mask)[0]

#         choose = rng.choice(
#             idx,
#             size=points_per_finger,
#             replace=(len(idx) < points_per_finger),
#         )

#         all_pts.append(points[choose])
#         all_ids.append(np.full(points_per_finger, fid, dtype=np.int32))

#         print(
#             f"[INFO] finger {fid}: center_idx={int(center_idx)}, "
#             f"coverage={coverage:.3f}, theoretical_patch={n_patch}, "
#             f"reachable_patch={n_reach}, sampled={points_per_finger}"
#         )

#     pts = np.vstack(all_pts).astype(np.float32)
#     ids = np.concatenate(all_ids).astype(np.int32)

#     return pts, ids


# # =========================================================
# # 11. visualization
# # =========================================================
# def visualize(points, finger_points, finger_ids, center_ids):
#     geometries = []

#     bg_n = min(50000, len(points))
#     bg_idx = np.random.choice(len(points), bg_n, replace=False)

#     bg = o3d.geometry.PointCloud()
#     bg.points = o3d.utility.Vector3dVector(points[bg_idx].astype(np.float64))
#     bg.paint_uniform_color([0.8, 0.8, 0.8])
#     geometries.append(bg)

#     colors = [
#         [1.0, 0.0, 0.0],
#         [0.0, 0.0, 1.0],
#         [0.0, 1.0, 0.0],
#         [1.0, 0.5, 0.0],
#         [1.0, 0.0, 1.0],
#     ]

#     for fid in range(5):
#         pts = finger_points[finger_ids == fid]
#         p = o3d.geometry.PointCloud()
#         p.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
#         p.paint_uniform_color(colors[fid])
#         geometries.append(p)

#     for cidx in center_ids:
#         s = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
#         s.translate(points[int(cidx)].astype(np.float64))
#         s.paint_uniform_color([0.0, 0.0, 0.0])
#         geometries.append(s)

#     o3d.visualization.draw_geometries(
#         geometries,
#         window_name="Deterministic Tactile Sampling with Global Beam Reachability",
#         point_show_normal=False,
#     )


# # =========================================================
# # 12. main pipeline
# # =========================================================
# def process_single_obj(
#     obj_path,
#     sample_point_count=200000,
#     num_rays=20000,
#     beam_radius=0.02,
#     rounds=5,
#     start_ratio=0.12,
#     end_ratio=0.03,
#     thickness_ratio=0.01,
#     points_per_finger=3000,
#     normal_angle_deg=28.0,
#     sample_seed=0,
#     ray_seed=0,
#     center_seed=0,
#     final_nonoverlap_safety=2.0,
#     min_reachable_coverage=0.5,
#     min_patch_points=150,
#     min_final_points=80,
# ):
#     # -----------------------------
#     # load and normalize mesh
#     # -----------------------------
#     mesh = trimesh.load(obj_path, force="mesh")
#     mesh.process(validate=True)
#     mesh = scale_to_unit_sphere(mesh)

#     # -----------------------------
#     # sample surface points
#     # -----------------------------
#     spc = sample_from_mesh(mesh, n=sample_point_count, seed=sample_seed)
#     diag = bbox_diag(spc.points)

#     # -----------------------------
#     # ray casting
#     # -----------------------------
#     candidate_ids, hit_points = raycast_outer_hits(
#         mesh=mesh,
#         points=spc.points,
#         num_rays=num_rays,
#         seed=ray_seed,
#     )

#     # -----------------------------
#     # global reachable region = union of all beams
#     # -----------------------------
#     reachable_mask = build_global_reachable_mask(
#         points=spc.points,
#         hit_points=hit_points,
#         beam_radius=beam_radius,
#     )

#     # -----------------------------
#     # compute largest/smallest round parameters
#     # -----------------------------
#     largest_radius = float(start_ratio) * diag
#     smallest_radius = float(end_ratio) * diag
#     thickness = float(thickness_ratio) * diag

#     # final-round non-overlap
#     min_center_dist = float(final_nonoverlap_safety) * smallest_radius

#     # -----------------------------
#     # fixed centers:
#     # must satisfy largest-round coverage constraint
#     # -----------------------------
#     center_ids = sample_valid_fixed_centers(
#         points=spc.points,
#         normals=spc.normals,
#         candidate_ids=candidate_ids,
#         reachable_mask=reachable_mask,
#         largest_radius=largest_radius,
#         thickness=thickness,
#         min_center_dist=min_center_dist,
#         num_centers=5,
#         normal_angle_deg=normal_angle_deg,
#         min_patch_points=min_patch_points,
#         min_final_points=min_final_points,
#         min_reachable_coverage=min_reachable_coverage,
#         seed=center_seed,
#     )

#     print("[INFO] fixed center ids:", center_ids.tolist())

#     # -----------------------------
#     # multi-round shrinking
#     # -----------------------------
#     radius_schedule = np.linspace(start_ratio, end_ratio, rounds)

#     for rid, radius_ratio in enumerate(radius_schedule):
#         radius = float(radius_ratio) * diag

#         print(f"\n[ROUND {rid}] radius_ratio = {radius_ratio:.6f}")

#         finger_points, finger_ids = tactile_sampling_round(
#             spc=spc,
#             reachable_mask=reachable_mask,
#             center_ids=center_ids,
#             radius=radius,
#             thickness=thickness,
#             points_per_finger=points_per_finger,
#             normal_angle_deg=normal_angle_deg,
#             min_patch_points=min_patch_points,
#             min_final_points=min_final_points,
#             min_reachable_coverage=min_reachable_coverage,
#             seed=rid,
#         )

#         visualize(
#             points=spc.points,
#             finger_points=finger_points,
#             finger_ids=finger_ids,
#             center_ids=center_ids,
#         )


# # =========================================================
# # 13. main
# # =========================================================
# if __name__ == "__main__":
#     OBJ_PATH = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/sofa/train_obj/sofa_0002.obj"

#     # process_single_obj(
#     #     obj_path=OBJ_PATH,
#     #     sample_point_count=200000,
#     #     num_rays=20000,
#     #     beam_radius=0.02,
#     #     rounds=5,
#     #     start_ratio=0.12,
#     #     end_ratio=0.03,
#     #     thickness_ratio=0.01,
#     #     points_per_finger=3000,
#     #     normal_angle_deg=28.0,
#     #     sample_seed=0,
#     #     ray_seed=0,
#     #     center_seed=0,
#     #     final_nonoverlap_safety=2.0,
#     #     min_reachable_coverage=0.5,
#     #     min_patch_points=150,
#     #     min_final_points=80,
#     # )

#     process_single_obj(
#     obj_path=OBJ_PATH,
#     sample_point_count=200000,
#     num_rays=20000,
#     beam_radius=0.02,
#     rounds=5,
#     start_ratio=0.12,
#     end_ratio=0.03,
#     thickness_ratio=0.01,
#     points_per_finger=3000,
#     normal_angle_deg=28.0,
# )







# 不设置固定seed，随机采样点

# import numpy as np
# import trimesh
# import open3d as o3d
# from scipy.spatial import cKDTree


# # =========================================================
# # 1. mesh normalization
# # =========================================================
# def scale_to_unit_sphere(mesh):
#     if isinstance(mesh, trimesh.Scene):
#         mesh = mesh.dump().sum()

#     vertices = mesh.vertices - mesh.bounding_box.centroid
#     distances = np.linalg.norm(vertices, axis=1)
#     max_dist = np.max(distances)

#     if max_dist <= 1e-12:
#         raise ValueError("Degenerate mesh: max distance is zero.")

#     vertices = vertices / max_dist
#     return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)


# # =========================================================
# # 2. surface point cloud container
# # =========================================================
# class SurfacePointCloud:
#     def __init__(self, mesh, points, normals):
#         self.mesh = mesh
#         self.points = points
#         self.normals = normals


# # =========================================================
# # 3. sample mesh surface  (随机版本)
# # =========================================================
# def sample_from_mesh(mesh, n=200000):

#     points, face_idx = mesh.sample(n, return_index=True)

#     normals = mesh.face_normals[face_idx].astype(np.float32)
#     norm = np.linalg.norm(normals, axis=1, keepdims=True)
#     normals = normals / np.clip(norm, 1e-12, None)

#     return SurfacePointCloud(
#         mesh=mesh,
#         points=points.astype(np.float32),
#         normals=normals.astype(np.float32),
#     )


# # =========================================================
# # bbox diagonal
# # =========================================================
# def bbox_diag(points):
#     mn = points.min(axis=0)
#     mx = points.max(axis=0)
#     return float(np.linalg.norm(mx - mn))


# # =========================================================
# # ray casting (随机方向)
# # =========================================================
# def raycast_outer_hits(mesh, points, num_rays=20000):

#     print("[INFO] ray casting")

#     legacy = o3d.geometry.TriangleMesh(
#         o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
#         o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
#     )

#     mesh_o3d = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
#     scene = o3d.t.geometry.RaycastingScene()
#     scene.add_triangles(mesh_o3d)

#     rng = np.random.default_rng()

#     dirs = rng.normal(size=(num_rays, 3))
#     dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

#     origins = dirs * 3.0
#     directions = -dirs

#     rays = np.concatenate([origins, directions], axis=1).astype(np.float32)
#     rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

#     ans = scene.cast_rays(rays)
#     t_hit = ans["t_hit"].numpy()

#     valid = np.isfinite(t_hit)
#     hit_points = origins[valid] + directions[valid] * t_hit[valid][:, None]

#     print("[INFO] raw ray hits:", len(hit_points))

#     tree = cKDTree(points)
#     _, ids = tree.query(hit_points, k=1)

#     candidate_ids = np.unique(ids.astype(np.int64))

#     print("[INFO] unique outer surface candidates:", len(candidate_ids))

#     return candidate_ids, hit_points.astype(np.float32)


# # =========================================================
# # reachable region (所有beam并集)
# # =========================================================
# def build_global_reachable_mask(points, hit_points, beam_radius):

#     print("[INFO] building reachable region")

#     hit_tree = cKDTree(hit_points)

#     dist, _ = hit_tree.query(points, k=1, distance_upper_bound=beam_radius)

#     reachable_mask = np.isfinite(dist)

#     print("[INFO] reachable points:", reachable_mask.sum())

#     return reachable_mask


# # =========================================================
# # tangent patch
# # =========================================================
# def extract_theoretical_patch(points, normals, center_idx,
#                               radius, thickness,
#                               normal_angle_deg=28,
#                               min_patch_points=150):

#     c = points[center_idx]
#     n = normals[center_idx]

#     v = points - c

#     height = v @ n

#     v_plane = v - height[:,None]*n

#     plane_dist = np.linalg.norm(v_plane, axis=1)

#     plane_mask = plane_dist <= radius
#     thickness_mask = np.abs(height) <= thickness

#     cos_th = np.cos(np.deg2rad(normal_angle_deg))
#     normal_mask = (normals @ n) >= cos_th

#     mask = plane_mask & thickness_mask & normal_mask

#     if mask.sum() < min_patch_points:
#         return None

#     return mask


# # =========================================================
# # patch + reachable constraint
# # =========================================================
# def extract_patch_with_global_reachability(
#         points,
#         normals,
#         reachable_mask,
#         center_idx,
#         radius,
#         thickness,
#         normal_angle_deg=28,
#         min_patch_points=150,
#         min_final_points=80,
#         min_reachable_coverage=0.2):

#     patch_mask = extract_theoretical_patch(
#         points,
#         normals,
#         center_idx,
#         radius,
#         thickness,
#         normal_angle_deg,
#         min_patch_points
#     )

#     if patch_mask is None:
#         return None,0,0,0

#     patch_ids = np.where(patch_mask)[0]

#     reachable_ids = patch_ids[reachable_mask[patch_ids]]

#     coverage = len(reachable_ids)/len(patch_ids)

#     if coverage < min_reachable_coverage:
#         return None,coverage,len(patch_ids),len(reachable_ids)

#     if len(reachable_ids) < min_final_points:
#         return None,coverage,len(patch_ids),len(reachable_ids)

#     final_mask = np.zeros(len(points),dtype=bool)
#     final_mask[reachable_ids] = True

#     return final_mask,coverage,len(patch_ids),len(reachable_ids)


# # =========================================================
# # center sampling (随机版本)
# # =========================================================
# def sample_valid_fixed_centers(points,
#                                normals,
#                                candidate_ids,
#                                reachable_mask,
#                                largest_radius,
#                                thickness,
#                                min_center_dist,
#                                num_centers=5):

#     rng = np.random.default_rng()

#     centers=[]

#     while len(centers)<num_centers:

#         idx=int(rng.choice(candidate_ids))

#         ok=True

#         for c in centers:
#             if np.linalg.norm(points[idx]-points[c])<min_center_dist:
#                 ok=False
#                 break

#         if ok:
#             centers.append(idx)

#     return np.array(centers)


# # =========================================================
# # tactile sampling round
# # =========================================================
# def tactile_sampling_round(spc,
#                            reachable_mask,
#                            center_ids,
#                            radius,
#                            thickness,
#                            points_per_finger=3000):

#     points=spc.points
#     normals=spc.normals

#     rng=np.random.default_rng()

#     all_pts=[]
#     all_ids=[]

#     for fid,center_idx in enumerate(center_ids):

#         mask,coverage,n_patch,n_reach = extract_patch_with_global_reachability(
#             points,
#             normals,
#             reachable_mask,
#             center_idx,
#             radius,
#             thickness
#         )

#         if mask is None:
#             continue

#         idx=np.where(mask)[0]

#         choose=rng.choice(idx,
#                           points_per_finger,
#                           replace=len(idx)<points_per_finger)

#         all_pts.append(points[choose])
#         all_ids.append(np.full(points_per_finger,fid))

#     pts=np.vstack(all_pts).astype(np.float32)
#     ids=np.concatenate(all_ids).astype(np.int32)

#     return pts,ids


# # =========================================================
# # visualization
# # =========================================================
# def visualize(points,finger_points,finger_ids,center_ids):

#     geometries=[]

#     idx=np.random.choice(len(points),50000,replace=False)

#     bg=o3d.geometry.PointCloud()
#     bg.points=o3d.utility.Vector3dVector(points[idx])
#     bg.paint_uniform_color([0.8,0.8,0.8])

#     geometries.append(bg)

#     colors=[
#         [1,0,0],
#         [0,0,1],
#         [0,1,0],
#         [1,0.5,0],
#         [1,0,1]
#     ]

#     for fid in range(5):

#         pts=finger_points[finger_ids==fid]

#         p=o3d.geometry.PointCloud()
#         p.points=o3d.utility.Vector3dVector(pts)
#         p.paint_uniform_color(colors[fid])

#         geometries.append(p)

#     for cidx in center_ids:

#         s=o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
#         s.translate(points[cidx])
#         s.paint_uniform_color([0,0,0])

#         geometries.append(s)

#     o3d.visualization.draw_geometries(geometries)


# # =========================================================
# # main
# # =========================================================
# def process_single_obj(obj_path):

#     mesh=trimesh.load(obj_path,force="mesh")
#     mesh.process()

#     mesh=scale_to_unit_sphere(mesh)

#     spc=sample_from_mesh(mesh)

#     diag=bbox_diag(spc.points)

#     candidate_ids,hit_points = raycast_outer_hits(mesh,spc.points)

#     reachable_mask=build_global_reachable_mask(
#         spc.points,
#         hit_points,
#         beam_radius=0.02
#     )

#     largest_radius=0.12*diag
#     smallest_radius=0.03*diag

#     thickness=0.01*diag

#     center_ids=sample_valid_fixed_centers(
#         spc.points,
#         spc.normals,
#         candidate_ids,
#         reachable_mask,
#         largest_radius,
#         thickness,
#         min_center_dist=2*smallest_radius
#     )

#     radius_schedule=np.linspace(0.12,0.03,5)

#     for rid,ratio in enumerate(radius_schedule):

#         radius=ratio*diag

#         print("\nRound",rid)

#         finger_points,finger_ids = tactile_sampling_round(
#             spc,
#             reachable_mask,
#             center_ids,
#             radius,
#             thickness
#         )

#         visualize(spc.points,finger_points,finger_ids,center_ids)


# # =========================================================
# # run
# # =========================================================
# if __name__=="__main__":

#     OBJ_PATH=r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/airplane/train_obj/airplane_0002.obj"

#     process_single_obj(OBJ_PATH)







# # 加入几何遮挡限制

# import numpy as np
# import trimesh
# import open3d as o3d
# from scipy.spatial import cKDTree


# # =========================================================
# # 1. mesh normalization
# # =========================================================
# def scale_to_unit_sphere(mesh):
#     if isinstance(mesh, trimesh.Scene):
#         mesh = mesh.dump().sum()

#     vertices = mesh.vertices - mesh.bounding_box.centroid
#     distances = np.linalg.norm(vertices, axis=1)
#     max_dist = np.max(distances)

#     if max_dist <= 1e-12:
#         raise ValueError("Degenerate mesh: max distance is zero.")

#     vertices = vertices / max_dist
#     return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)


# # =========================================================
# # 2. surface point cloud container
# # =========================================================
# class SurfacePointCloud:
#     def __init__(self, mesh, points, normals):
#         self.mesh = mesh
#         self.points = points
#         self.normals = normals


# # =========================================================
# # 3. sample mesh surface
# # =========================================================
# def sample_from_mesh(mesh, n=200000):
#     points, face_idx = mesh.sample(n, return_index=True)

#     normals = mesh.face_normals[face_idx].astype(np.float32)
#     norm = np.linalg.norm(normals, axis=1, keepdims=True)
#     normals = normals / np.clip(norm, 1e-12, None)

#     return SurfacePointCloud(
#         mesh=mesh,
#         points=points.astype(np.float32),
#         normals=normals.astype(np.float32),
#     )


# # =========================================================
# # bbox diagonal
# # =========================================================
# def bbox_diag(points):
#     mn = points.min(axis=0)
#     mx = points.max(axis=0)
#     return float(np.linalg.norm(mx - mn))


# # =========================================================
# # build raycasting scene
# # =========================================================
# def build_raycast_scene(mesh):
#     legacy = o3d.geometry.TriangleMesh(
#         o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
#         o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
#     )

#     mesh_o3d = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
#     scene = o3d.t.geometry.RaycastingScene()
#     scene.add_triangles(mesh_o3d)
#     return scene


# # =========================================================
# # ray casting (global outer-hit candidates)
# # =========================================================
# def raycast_outer_hits(mesh, points, num_rays=20000):
#     print("[INFO] ray casting")

#     scene = build_raycast_scene(mesh)

#     rng = np.random.default_rng()

#     dirs = rng.normal(size=(num_rays, 3))
#     dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

#     origins = dirs * 3.0
#     directions = -dirs

#     rays = np.concatenate([origins, directions], axis=1).astype(np.float32)
#     rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

#     ans = scene.cast_rays(rays)
#     t_hit = ans["t_hit"].numpy()

#     valid = np.isfinite(t_hit)
#     hit_points = origins[valid] + directions[valid] * t_hit[valid][:, None]

#     print("[INFO] raw ray hits:", len(hit_points))

#     tree = cKDTree(points)
#     _, ids = tree.query(hit_points, k=1)

#     candidate_ids = np.unique(ids.astype(np.int64))

#     print("[INFO] unique outer surface candidates:", len(candidate_ids))

#     return candidate_ids, hit_points.astype(np.float32)


# # =========================================================
# # reachable region (global)
# # =========================================================
# def build_global_reachable_mask(points, hit_points, beam_radius):
#     print("[INFO] building reachable region")

#     hit_tree = cKDTree(hit_points)
#     dist, _ = hit_tree.query(points, k=1, distance_upper_bound=beam_radius)

#     reachable_mask = np.isfinite(dist)

#     print("[INFO] reachable points:", int(reachable_mask.sum()))
#     return reachable_mask


# # =========================================================
# # tangent patch
# # =========================================================
# def extract_theoretical_patch(points, normals, center_idx,
#                               radius, thickness,
#                               normal_angle_deg=28,
#                               min_patch_points=150):
#     c = points[center_idx]
#     n = normals[center_idx]

#     v = points - c
#     height = v @ n
#     v_plane = v - height[:, None] * n
#     plane_dist = np.linalg.norm(v_plane, axis=1)

#     plane_mask = plane_dist <= radius
#     thickness_mask = np.abs(height) <= thickness

#     cos_th = np.cos(np.deg2rad(normal_angle_deg))
#     normal_mask = (normals @ n) >= cos_th

#     mask = plane_mask & thickness_mask & normal_mask

#     if mask.sum() < min_patch_points:
#         return None

#     return mask


# # =========================================================
# # local visibility filter
# # =========================================================
# def filter_visible_points(scene,
#                           center,
#                           center_normal,
#                           candidate_points,
#                           eps=2e-3,
#                           hit_tol=2e-3):
#     """
#     从中心点沿法向外偏移 eps，向候选点发射射线。
#     若第一命中点距离 ~= 候选点距离，则认为可见；
#     否则说明中间被 mesh 挡住，不可见。
#     """
#     if len(candidate_points) == 0:
#         return np.zeros(0, dtype=bool)

#     origin = center + center_normal * eps

#     vec = candidate_points - origin[None, :]
#     dist = np.linalg.norm(vec, axis=1)

#     valid_dir = dist > 1e-12
#     visible = np.zeros(len(candidate_points), dtype=bool)

#     if not np.any(valid_dir):
#         return visible

#     dirs = vec[valid_dir] / dist[valid_dir][:, None]
#     origins = np.repeat(origin[None, :], len(dirs), axis=0)

#     rays = np.concatenate([origins, dirs], axis=1).astype(np.float32)
#     rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

#     ans = scene.cast_rays(rays)
#     t_hit = ans["t_hit"].numpy()

#     finite_hit = np.isfinite(t_hit)

#     valid_indices = np.where(valid_dir)[0]

#     # 如果第一命中距离接近目标点距离，则可见
#     ok = np.zeros(len(dirs), dtype=bool)
#     ok[finite_hit] = np.abs(t_hit[finite_hit] - dist[valid_dir][finite_hit]) <= hit_tol

#     visible[valid_indices] = ok
#     return visible


# # =========================================================
# # patch + reachable + local visibility
# # =========================================================
# def extract_patch_with_visibility(
#         scene,
#         points,
#         normals,
#         reachable_mask,
#         center_idx,
#         radius,
#         thickness,
#         normal_angle_deg=28,
#         min_patch_points=150,
#         min_final_points=80,
#         min_reachable_coverage=0.2,
#         min_visible_coverage=0.3,
#         eps=1e-3,
#         hit_tol=3e-3):

#     patch_mask = extract_theoretical_patch(
#         points,
#         normals,
#         center_idx,
#         radius,
#         thickness,
#         normal_angle_deg,
#         min_patch_points
#     )

#     if patch_mask is None:
#         return None, {
#             "reason": "theoretical_patch_too_small",
#             "coverage_reachable": 0.0,
#             "coverage_visible": 0.0,
#             "n_patch": 0,
#             "n_reachable": 0,
#             "n_visible": 0
#         }

#     patch_ids = np.where(patch_mask)[0]

#     reachable_ids = patch_ids[reachable_mask[patch_ids]]
#     coverage_reachable = len(reachable_ids) / max(len(patch_ids), 1)

#     if coverage_reachable < min_reachable_coverage:
#         return None, {
#             "reason": "reachable_coverage_too_low",
#             "coverage_reachable": coverage_reachable,
#             "coverage_visible": 0.0,
#             "n_patch": len(patch_ids),
#             "n_reachable": len(reachable_ids),
#             "n_visible": 0
#         }

#     if len(reachable_ids) < min_final_points:
#         return None, {
#             "reason": "reachable_points_too_few",
#             "coverage_reachable": coverage_reachable,
#             "coverage_visible": 0.0,
#             "n_patch": len(patch_ids),
#             "n_reachable": len(reachable_ids),
#             "n_visible": 0
#         }

#     center = points[center_idx]
#     center_normal = normals[center_idx]
#     candidate_points = points[reachable_ids]

#     visible_local = filter_visible_points(
#         scene=scene,
#         center=center,
#         center_normal=center_normal,
#         candidate_points=candidate_points,
#         eps=eps,
#         hit_tol=hit_tol
#     )

#     visible_ids = reachable_ids[visible_local]
#     coverage_visible = len(visible_ids) / max(len(reachable_ids), 1)

#     if coverage_visible < min_visible_coverage:
#         return None, {
#             "reason": "visible_coverage_too_low",
#             "coverage_reachable": coverage_reachable,
#             "coverage_visible": coverage_visible,
#             "n_patch": len(patch_ids),
#             "n_reachable": len(reachable_ids),
#             "n_visible": len(visible_ids)
#         }

#     if len(visible_ids) < min_final_points:
#         return None, {
#             "reason": "visible_points_too_few",
#             "coverage_reachable": coverage_reachable,
#             "coverage_visible": coverage_visible,
#             "n_patch": len(patch_ids),
#             "n_reachable": len(reachable_ids),
#             "n_visible": len(visible_ids)
#         }

#     final_mask = np.zeros(len(points), dtype=bool)
#     final_mask[visible_ids] = True

#     return final_mask, {
#         "reason": "ok",
#         "coverage_reachable": coverage_reachable,
#         "coverage_visible": coverage_visible,
#         "n_patch": len(patch_ids),
#         "n_reachable": len(reachable_ids),
#         "n_visible": len(visible_ids)
#     }


# # =========================================================
# # center sampling
# # =========================================================
# def sample_valid_fixed_centers(scene,
#                                points,
#                                normals,
#                                candidate_ids,
#                                reachable_mask,
#                                largest_radius,
#                                thickness,
#                                min_center_dist,
#                                num_centers=5,
#                                max_trials=5000):
#     rng = np.random.default_rng()
#     centers = []

#     trials = 0
#     while len(centers) < num_centers and trials < max_trials:
#         trials += 1

#         idx = int(rng.choice(candidate_ids))

#         ok_dist = True
#         for c in centers:
#             if np.linalg.norm(points[idx] - points[c]) < min_center_dist:
#                 ok_dist = False
#                 break

#         if not ok_dist:
#             continue

#         # 要求最大半径下也能提取出一个几何上合理的 patch
#         mask, info = extract_patch_with_visibility(
#             scene=scene,
#             points=points,
#             normals=normals,
#             reachable_mask=reachable_mask,
#             center_idx=idx,
#             radius=largest_radius,
#             thickness=thickness
#         )

#         if mask is None:
#             continue

#         centers.append(idx)
#         print(
#             f"[CENTER] accept center {len(centers)-1}: idx={idx}, "
#             f"patch={info['n_patch']}, reachable={info['n_reachable']}, visible={info['n_visible']}"
#         )

#     if len(centers) < num_centers:
#         raise RuntimeError(
#             f"Only found {len(centers)} valid centers, need {num_centers}. "
#             f"Try increasing sample count or relaxing thresholds."
#         )

#     return np.array(centers, dtype=np.int64)


# # =========================================================
# # tactile sampling round
# # =========================================================
# def tactile_sampling_round(spc,
#                            scene,
#                            reachable_mask,
#                            center_ids,
#                            radius,
#                            thickness,
#                            points_per_finger=3000):
#     points = spc.points
#     normals = spc.normals
#     rng = np.random.default_rng()

#     all_pts = []
#     all_ids = []

#     for fid, center_idx in enumerate(center_ids):
#         mask, info = extract_patch_with_visibility(
#             scene=scene,
#             points=points,
#             normals=normals,
#             reachable_mask=reachable_mask,
#             center_idx=center_idx,
#             radius=radius,
#             thickness=thickness
#         )

#         if mask is None:
#             print(
#                 f"[Finger {fid}] NO PATCH | reason={info['reason']} | "
#                 f"patch={info['n_patch']} | reachable={info['n_reachable']} "
#                 f"({info['coverage_reachable']:.3f}) | "
#                 f"visible={info['n_visible']} ({info['coverage_visible']:.3f})"
#             )
#             continue

#         idx = np.where(mask)[0]

#         choose = rng.choice(
#             idx,
#             points_per_finger,
#             replace=len(idx) < points_per_finger
#         )

#         all_pts.append(points[choose])
#         all_ids.append(np.full(points_per_finger, fid, dtype=np.int32))

#         print(
#             f"[Finger {fid}] OK | patch={info['n_patch']} | "
#             f"reachable={info['n_reachable']} ({info['coverage_reachable']:.3f}) | "
#             f"visible={info['n_visible']} ({info['coverage_visible']:.3f}) | "
#             f"sampled={points_per_finger}"
#         )

#     if len(all_pts) == 0:
#         return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)

#     pts = np.vstack(all_pts).astype(np.float32)
#     ids = np.concatenate(all_ids).astype(np.int32)

#     return pts, ids


# # =========================================================
# # visualization
# # =========================================================
# def visualize(points, finger_points, finger_ids, center_ids):
#     geometries = []

#     n_bg = min(50000, len(points))
#     idx = np.random.choice(len(points), n_bg, replace=False)

#     bg = o3d.geometry.PointCloud()
#     bg.points = o3d.utility.Vector3dVector(points[idx])
#     bg.paint_uniform_color([0.85, 0.85, 0.85])
#     geometries.append(bg)

#     colors = [
#         [1, 0, 0],
#         [0, 0, 1],
#         [0, 1, 0],
#         [1, 0.5, 0],
#         [1, 0, 1]
#     ]

#     for fid in range(5):
#         pts = finger_points[finger_ids == fid]
#         if len(pts) == 0:
#             continue

#         p = o3d.geometry.PointCloud()
#         p.points = o3d.utility.Vector3dVector(pts)
#         p.paint_uniform_color(colors[fid])
#         geometries.append(p)

#     for cidx in center_ids:
#         s = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
#         s.translate(points[cidx])
#         s.paint_uniform_color([0, 0, 0])
#         geometries.append(s)

#     o3d.visualization.draw_geometries(geometries)


# # =========================================================
# # main
# # =========================================================
# def process_single_obj(obj_path):
#     mesh = trimesh.load(obj_path, force="mesh")
#     mesh.process()

#     mesh = scale_to_unit_sphere(mesh)
#     spc = sample_from_mesh(mesh)

#     diag = bbox_diag(spc.points)

#     # 全局 raycast 候选
#     candidate_ids, hit_points = raycast_outer_hits(mesh, spc.points)

#     reachable_mask = build_global_reachable_mask(
#         spc.points,
#         hit_points,
#         beam_radius=0.1
#     )

#     # 局部遮挡检测 scene
#     scene = build_raycast_scene(mesh)

#     largest_radius = 0.12 * diag
#     smallest_radius = 0.03 * diag
#     thickness = 0.01 * diag

#     center_ids = sample_valid_fixed_centers(
#         scene=scene,
#         points=spc.points,
#         normals=spc.normals,
#         candidate_ids=candidate_ids,
#         reachable_mask=reachable_mask,
#         largest_radius=largest_radius,
#         thickness=thickness,
#         min_center_dist=2 * smallest_radius
#     )

#     print("[INFO] fixed center ids:", center_ids.tolist())

#     radius_schedule = np.linspace(0.12, 0.03, 5)

#     for rid, ratio in enumerate(radius_schedule):
#         radius = ratio * diag

#         print(f"\n[ROUND {rid}] radius_ratio={ratio:.4f}, radius={radius:.6f}")

#         finger_points, finger_ids = tactile_sampling_round(
#             spc=spc,
#             scene=scene,
#             reachable_mask=reachable_mask,
#             center_ids=center_ids,
#             radius=radius,
#             thickness=thickness
#         )

#         visualize(spc.points, finger_points, finger_ids, center_ids)


# # =========================================================
# # run
# # =========================================================
# if __name__ == "__main__":
#     OBJ_PATH = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/piano/train_obj/piano_0002.obj"
#     process_single_obj(OBJ_PATH)











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