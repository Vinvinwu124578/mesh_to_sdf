# import os
# import numpy as np
# import trimesh
# import open3d as o3d

# from scipy.spatial import cKDTree
# from mesh_to_sdf import mesh_to_sdf


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
# # 3. dense surface sampling for tactile generation
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
# # 4. bbox diagonal
# # =========================================================
# def bbox_diag(points):
#     mn = points.min(axis=0)
#     mx = points.max(axis=0)
#     return float(np.linalg.norm(mx - mn))


# # =========================================================
# # 5. build raycasting scene
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
# # 6. ray casting (global outer-hit candidates)
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
# # 7. reachable region (global)
# # =========================================================
# def build_global_reachable_mask(points, hit_points, beam_radius):
#     print("[INFO] building reachable region")

#     hit_tree = cKDTree(hit_points)
#     dist, _ = hit_tree.query(points, k=1, distance_upper_bound=beam_radius)

#     reachable_mask = np.isfinite(dist)

#     print("[INFO] reachable points:", int(reachable_mask.sum()))
#     return reachable_mask


# # =========================================================
# # 8. tangent patch
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
# # 9. local visibility filter
# # =========================================================
# def filter_visible_points(scene,
#                           center,
#                           center_normal,
#                           candidate_points,
#                           eps=2e-3,
#                           hit_tol=2e-3):
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

#     ok = np.zeros(len(dirs), dtype=bool)
#     ok[finite_hit] = np.abs(t_hit[finite_hit] - dist[valid_dir][finite_hit]) <= hit_tol

#     visible[valid_indices] = ok
#     return visible


# # =========================================================
# # 10. patch + reachable + local visibility
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
# # 11. edge-aware soft contact probability
# # =========================================================
# def estimate_edge_deformation_factor(points,
#                                      normals,
#                                      center_idx,
#                                      radius,
#                                      edge_neighbor_ratio=0.35):
#     c = points[center_idx]
#     n0 = normals[center_idx]

#     d = np.linalg.norm(points - c[None, :], axis=1)
#     local_mask = d <= (edge_neighbor_ratio * radius)
#     local_ids = np.where(local_mask)[0]

#     if len(local_ids) < 20:
#         return 0.0

#     align = normals[local_ids] @ n0
#     align = np.clip(align, -1.0, 1.0)

#     normal_variation = 1.0 - np.mean(np.abs(align))
#     edge_factor = np.clip(normal_variation / 0.5, 0.0, 1.0)
#     return float(edge_factor)


# def compute_soft_contact_probability(points,
#                                      normals,
#                                      center_idx,
#                                      candidate_ids,
#                                      radius,
#                                      cross_surface_gain=0.35,
#                                      edge_neighbor_ratio=0.35):
#     if len(candidate_ids) == 0:
#         return np.zeros((0,), dtype=np.float64)

#     c = points[center_idx]
#     n0 = normals[center_idx]

#     pts = points[candidate_ids]
#     nrm = normals[candidate_ids]

#     v = pts - c[None, :]
#     dist = np.linalg.norm(v, axis=1)

#     radial = 1.0 - dist / max(radius, 1e-8)
#     radial = np.clip(radial, 0.0, None) ** 2

#     align = nrm @ n0
#     align_pos = np.clip(align, 0.0, 1.0)

#     edge_factor = estimate_edge_deformation_factor(
#         points=points,
#         normals=normals,
#         center_idx=center_idx,
#         radius=radius,
#         edge_neighbor_ratio=edge_neighbor_ratio
#     )

#     soft_cross = np.clip((align + 1.0) * 0.5, 0.0, 1.0)
#     soft_cross = soft_cross ** 2

#     prob = radial * (align_pos + cross_surface_gain * edge_factor * soft_cross)

#     if np.all(prob <= 1e-12):
#         prob = np.ones_like(prob, dtype=np.float64)

#     prob = prob.astype(np.float64)
#     prob /= prob.sum()

#     return prob


# # =========================================================
# # 12. center sampling
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
# # 13. tactile sampling round
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
#     all_center_ids = []

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

#         prob = compute_soft_contact_probability(
#             points=points,
#             normals=normals,
#             center_idx=center_idx,
#             candidate_ids=idx,
#             radius=radius,
#             cross_surface_gain=0.35,
#             edge_neighbor_ratio=0.35
#         )

#         choose = rng.choice(
#             idx,
#             points_per_finger,
#             replace=len(idx) < points_per_finger,
#             p=prob
#         )

#         all_pts.append(points[choose])
#         all_ids.append(np.full(points_per_finger, fid, dtype=np.int32))
#         all_center_ids.append(np.full(points_per_finger, center_idx, dtype=np.int32))

#         edge_factor = estimate_edge_deformation_factor(
#             points=points,
#             normals=normals,
#             center_idx=center_idx,
#             radius=radius,
#             edge_neighbor_ratio=0.35
#         )

#         print(
#             f"[Finger {fid}] OK | patch={info['n_patch']} | "
#             f"reachable={info['n_reachable']} ({info['coverage_reachable']:.3f}) | "
#             f"visible={info['n_visible']} ({info['coverage_visible']:.3f}) | "
#             f"edge_factor={edge_factor:.3f} | sampled={points_per_finger}"
#         )

#     if len(all_pts) == 0:
#         return (
#             np.zeros((0, 3), dtype=np.float32),
#             np.zeros((0,), dtype=np.int32),
#             np.zeros((0,), dtype=np.int32),
#         )

#     pts = np.vstack(all_pts).astype(np.float32)
#     ids = np.concatenate(all_ids).astype(np.int32)
#     center_ids_per_point = np.concatenate(all_center_ids).astype(np.int32)

#     return pts, ids, center_ids_per_point


# # =========================================================
# # 14. generate tactile touch points
# # =========================================================
# def generate_tactile_touch_points(
#     mesh,
#     surface_sample_n=200000,
#     num_rays=20000,
#     beam_radius=0.1,
#     rounds=5,
#     start_ratio=0.12,
#     end_ratio=0.03,
#     thickness_ratio=0.01,
#     points_per_finger=3000,
#     num_fingers=5,
# ):
#     spc = sample_from_mesh(mesh, n=surface_sample_n)
#     diag = bbox_diag(spc.points)

#     candidate_ids, hit_points = raycast_outer_hits(mesh, spc.points, num_rays=num_rays)

#     reachable_mask = build_global_reachable_mask(
#         spc.points,
#         hit_points,
#         beam_radius=beam_radius
#     )

#     scene = build_raycast_scene(mesh)

#     largest_radius = start_ratio * diag
#     smallest_radius = end_ratio * diag
#     thickness = thickness_ratio * diag

#     center_ids = sample_valid_fixed_centers(
#         scene=scene,
#         points=spc.points,
#         normals=spc.normals,
#         candidate_ids=candidate_ids,
#         reachable_mask=reachable_mask,
#         largest_radius=largest_radius,
#         thickness=thickness,
#         min_center_dist=2 * smallest_radius,
#         num_centers=num_fingers
#     )

#     print("[INFO] fixed center ids:", center_ids.tolist())

#     radius_schedule = np.linspace(start_ratio, end_ratio, rounds)

#     round_points = []
#     round_ids = []
#     finger_ids = []
#     center_ids_per_point = []

#     for rid, ratio in enumerate(radius_schedule):
#         radius = ratio * diag

#         print(f"\n[ROUND {rid}] radius_ratio={ratio:.4f}, radius={radius:.6f}")

#         pts, fids, cids = tactile_sampling_round(
#             spc=spc,
#             scene=scene,
#             reachable_mask=reachable_mask,
#             center_ids=center_ids,
#             radius=radius,
#             thickness=thickness,
#             points_per_finger=points_per_finger
#         )

#         if len(pts) == 0:
#             continue

#         round_points.append(pts)
#         round_ids.append(np.full(len(pts), rid, dtype=np.int32))
#         finger_ids.append(fids)
#         center_ids_per_point.append(cids)

#     if len(round_points) == 0:
#         raise RuntimeError("No tactile points were generated.")

#     touch_points = np.vstack(round_points).astype(np.float32)
#     touch_round_ids = np.concatenate(round_ids).astype(np.int32)
#     touch_finger_ids = np.concatenate(finger_ids).astype(np.int32)
#     touch_center_ids = np.concatenate(center_ids_per_point).astype(np.int32)
#     touch_centers = spc.points[center_ids].astype(np.float32)

#     return {
#         "touch_points": touch_points,
#         "touch_round_ids": touch_round_ids,
#         "touch_finger_ids": touch_finger_ids,
#         "touch_center_ids": touch_center_ids,
#         "touch_centers": touch_centers,
#     }


# # =========================================================
# # 15. Chou-style surface/query sampling
# # =========================================================
# def sample_training_points_chou_style(
#     mesh,
#     num_surface_points=235000,
#     sigma_large=0.005,
#     sigma_small=0.0005,
# ):
#     print("[INFO] sampling surface/query points in Chou-style")

#     surface_points, face_idx = mesh.sample(num_surface_points, return_index=True)
#     surface_points = surface_points.astype(np.float32)

#     surface_normals = mesh.face_normals[face_idx].astype(np.float32)
#     surface_normals /= np.clip(
#         np.linalg.norm(surface_normals, axis=1, keepdims=True),
#         1e-8,
#         None
#     )

#     noise_large = np.random.normal(
#         loc=0.0,
#         scale=sigma_large,
#         size=surface_points.shape
#     ).astype(np.float32)

#     noise_small = np.random.normal(
#         loc=0.0,
#         scale=sigma_small,
#         size=surface_points.shape
#     ).astype(np.float32)

#     query_points_large = surface_points + noise_large
#     query_points_small = surface_points + noise_small

#     query_points = np.concatenate(
#         [query_points_large, query_points_small],
#         axis=0
#     ).astype(np.float32)

#     print("[INFO] computing query_sdf ...")
#     query_sdf = mesh_to_sdf(mesh, query_points).astype(np.float32)

#     surface_sdf = np.zeros((num_surface_points,), dtype=np.float32)

#     return {
#         "surface_points": surface_points,
#         "surface_normals": surface_normals,
#         "surface_sdf": surface_sdf,
#         "query_points": query_points,
#         "query_sdf": query_sdf,
#     }


# # =========================================================
# # 16. process single obj -> npz
# # =========================================================
# def process_single_obj_to_npz(
#     obj_path,
#     out_path,
#     tactile_surface_sample_n=200000,
#     tactile_num_rays=20000,
#     tactile_beam_radius=0.1,
#     tactile_rounds=5,
#     tactile_start_ratio=0.12,
#     tactile_end_ratio=0.03,
#     tactile_thickness_ratio=0.01,
#     tactile_points_per_finger=3000,
#     tactile_num_fingers=5,
#     num_surface_points=235000,
#     sigma_large=0.005,
#     sigma_small=0.0005,
# ):
#     print("\n==================================================")
#     print("[PROCESS]", obj_path)
#     print("==================================================")

#     mesh = trimesh.load(obj_path, force="mesh")
#     mesh.process(validate=True)
#     mesh = scale_to_unit_sphere(mesh)

#     tactile_data = generate_tactile_touch_points(
#         mesh=mesh,
#         surface_sample_n=tactile_surface_sample_n,
#         num_rays=tactile_num_rays,
#         beam_radius=tactile_beam_radius,
#         rounds=tactile_rounds,
#         start_ratio=tactile_start_ratio,
#         end_ratio=tactile_end_ratio,
#         thickness_ratio=tactile_thickness_ratio,
#         points_per_finger=tactile_points_per_finger,
#         num_fingers=tactile_num_fingers,
#     )

#     shape_data = sample_training_points_chou_style(
#         mesh=mesh,
#         num_surface_points=num_surface_points,
#         sigma_large=sigma_large,
#         sigma_small=sigma_small,
#     )

#     os.makedirs(os.path.dirname(out_path), exist_ok=True)

#     np.savez_compressed(
#         out_path,
#         surface_points=shape_data["surface_points"],
#         surface_normals=shape_data["surface_normals"],
#         surface_sdf=shape_data["surface_sdf"],
#         touch_points=tactile_data["touch_points"],
#         touch_round_ids=tactile_data["touch_round_ids"],
#         touch_finger_ids=tactile_data["touch_finger_ids"],
#         touch_center_ids=tactile_data["touch_center_ids"],
#         touch_centers=tactile_data["touch_centers"],
#         query_points=shape_data["query_points"],
#         query_sdf=shape_data["query_sdf"],
#         mesh_name=np.array(os.path.basename(obj_path)),
#     )

#     print("[SAVED]", out_path)
#     print("surface_points:", shape_data["surface_points"].shape)
#     print("touch_points  :", tactile_data["touch_points"].shape)
#     print("query_points  :", shape_data["query_points"].shape)


# # =========================================================
# # 17. process one split of one category
# # =========================================================
# def process_split(
#     category_dir,
#     split="train",
#     max_objects=80,
#     output_folder_name=None,
# ):
#     obj_dir = os.path.join(category_dir, f"{split}_obj")

#     if output_folder_name is None:
#         output_folder_name = f"tactistruct_npz_{split}"

#     out_dir = os.path.join(category_dir, output_folder_name)

#     if not os.path.isdir(obj_dir):
#         print(f"[WARN] skip missing dir: {obj_dir}")
#         return

#     os.makedirs(out_dir, exist_ok=True)

#     obj_files = sorted([
#         f for f in os.listdir(obj_dir)
#         if f.lower().endswith(".obj")
#     ])[:max_objects]

#     print(f"[INFO] {split}: found {len(obj_files)} obj files in {obj_dir}")

#     for name in obj_files:
#         obj_path = os.path.join(obj_dir, name)
#         out_path = os.path.join(out_dir, name[:-4] + ".npz")

#         if os.path.exists(out_path):
#             print("[SKIP exists]", out_path)
#             continue

#         try:
#             process_single_obj_to_npz(
#                 obj_path=obj_path,
#                 out_path=out_path,
#             )
#         except Exception as e:
#             print("[FAILED]", obj_path)
#             print("Error:", e)


# # =========================================================
# # 18. process all categories
# # =========================================================
# def process_all_categories(
#     root_dir,
#     split="train",
#     max_objects_per_category=80,
#     category_names=None,
# ):
#     subdirs = sorted([
#         os.path.join(root_dir, d)
#         for d in os.listdir(root_dir)
#         if os.path.isdir(os.path.join(root_dir, d))
#     ])

#     if category_names is not None:
#         category_names = set(category_names)
#         subdirs = [d for d in subdirs if os.path.basename(d) in category_names]

#     if not subdirs:
#         print("[WARN] no category folders found under:", root_dir)
#         return

#     print(f"[INFO] found {len(subdirs)} category folders under root.")

#     for category_dir in subdirs:
#         category_name = os.path.basename(category_dir)
#         print(f"\n########## Processing category: {category_name} ##########")

#         process_split(
#             category_dir=category_dir,
#             split=split,
#             max_objects=max_objects_per_category,
#             output_folder_name=f"tactistruct_npz_{split}",
#         )


# # =========================================================
# # 19. main
# # =========================================================
# if __name__ == "__main__":
#     root_dir = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40"

#     # 单类别测试：
#     # process_all_categories(
#     #     root_dir=root_dir,
#     #     split="train",
#     #     max_objects_per_category=2,
#     #     category_names=["chair"]
#     # )

#     # 全部类别：
#     process_all_categories(
#         root_dir=root_dir,
#         split="train",
#         max_objects_per_category=80,
#         category_names=None
#     )

#     print("\nAll done.")







# import os
# import numpy as np
# import trimesh
# import open3d as o3d

# from scipy.spatial import cKDTree
# from mesh_to_sdf import mesh_to_sdf


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
# # 3. dense surface sampling for tactile generation
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
# # 4. bbox diagonal
# # =========================================================
# def bbox_diag(points):
#     mn = points.min(axis=0)
#     mx = points.max(axis=0)
#     return float(np.linalg.norm(mx - mn))


# # =========================================================
# # 5. build raycasting scene
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
# # 6. ray casting (global outer-hit candidates)
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
# # 7. reachable region (global)
# # =========================================================
# def build_global_reachable_mask(points, hit_points, beam_radius):
#     print("[INFO] building reachable region")

#     hit_tree = cKDTree(hit_points)
#     dist, _ = hit_tree.query(points, k=1, distance_upper_bound=beam_radius)

#     reachable_mask = np.isfinite(dist)

#     print("[INFO] reachable points:", int(reachable_mask.sum()))
#     return reachable_mask


# # =========================================================
# # 8. tangent patch
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
# # 9. local visibility filter
# # =========================================================
# def filter_visible_points(scene,
#                           center,
#                           center_normal,
#                           candidate_points,
#                           eps=2e-3,
#                           hit_tol=2e-3):
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

#     ok = np.zeros(len(dirs), dtype=bool)
#     ok[finite_hit] = np.abs(t_hit[finite_hit] - dist[valid_dir][finite_hit]) <= hit_tol

#     visible[valid_indices] = ok
#     return visible


# # =========================================================
# # 10. patch + reachable + local visibility
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
# # 11. edge-aware soft contact probability
# # =========================================================
# def estimate_edge_deformation_factor(points,
#                                      normals,
#                                      center_idx,
#                                      radius,
#                                      edge_neighbor_ratio=0.35):
#     c = points[center_idx]
#     n0 = normals[center_idx]

#     d = np.linalg.norm(points - c[None, :], axis=1)
#     local_mask = d <= (edge_neighbor_ratio * radius)
#     local_ids = np.where(local_mask)[0]

#     if len(local_ids) < 20:
#         return 0.0

#     align = normals[local_ids] @ n0
#     align = np.clip(align, -1.0, 1.0)

#     normal_variation = 1.0 - np.mean(np.abs(align))
#     edge_factor = np.clip(normal_variation / 0.5, 0.0, 1.0)
#     return float(edge_factor)


# def compute_soft_contact_probability(points,
#                                      normals,
#                                      center_idx,
#                                      candidate_ids,
#                                      radius,
#                                      cross_surface_gain=0.35,
#                                      edge_neighbor_ratio=0.35):
#     if len(candidate_ids) == 0:
#         return np.zeros((0,), dtype=np.float64)

#     c = points[center_idx]
#     n0 = normals[center_idx]

#     pts = points[candidate_ids]
#     nrm = normals[candidate_ids]

#     v = pts - c[None, :]
#     dist = np.linalg.norm(v, axis=1)

#     radial = 1.0 - dist / max(radius, 1e-8)
#     radial = np.clip(radial, 0.0, None) ** 2

#     align = nrm @ n0
#     align_pos = np.clip(align, 0.0, 1.0)

#     edge_factor = estimate_edge_deformation_factor(
#         points=points,
#         normals=normals,
#         center_idx=center_idx,
#         radius=radius,
#         edge_neighbor_ratio=edge_neighbor_ratio
#     )

#     soft_cross = np.clip((align + 1.0) * 0.5, 0.0, 1.0)
#     soft_cross = soft_cross ** 2

#     prob = radial * (align_pos + cross_surface_gain * edge_factor * soft_cross)

#     if np.all(prob <= 1e-12):
#         prob = np.ones_like(prob, dtype=np.float64)

#     prob = prob.astype(np.float64)
#     prob /= prob.sum()

#     return prob


# # =========================================================
# # 12. center sampling
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
# # 13. tactile sampling round
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
#     all_center_ids = []

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

#         prob = compute_soft_contact_probability(
#             points=points,
#             normals=normals,
#             center_idx=center_idx,
#             candidate_ids=idx,
#             radius=radius,
#             cross_surface_gain=0.35,
#             edge_neighbor_ratio=0.35
#         )

#         choose = rng.choice(
#             idx,
#             points_per_finger,
#             replace=len(idx) < points_per_finger,
#             p=prob
#         )

#         all_pts.append(points[choose])
#         all_ids.append(np.full(points_per_finger, fid, dtype=np.int32))
#         all_center_ids.append(np.full(points_per_finger, center_idx, dtype=np.int32))

#         edge_factor = estimate_edge_deformation_factor(
#             points=points,
#             normals=normals,
#             center_idx=center_idx,
#             radius=radius,
#             edge_neighbor_ratio=0.35
#         )

#         print(
#             f"[Finger {fid}] OK | patch={info['n_patch']} | "
#             f"reachable={info['n_reachable']} ({info['coverage_reachable']:.3f}) | "
#             f"visible={info['n_visible']} ({info['coverage_visible']:.3f}) | "
#             f"edge_factor={edge_factor:.3f} | sampled={points_per_finger}"
#         )

#     if len(all_pts) == 0:
#         return (
#             np.zeros((0, 3), dtype=np.float32),
#             np.zeros((0,), dtype=np.int32),
#             np.zeros((0,), dtype=np.int32),
#         )

#     pts = np.vstack(all_pts).astype(np.float32)
#     ids = np.concatenate(all_ids).astype(np.int32)
#     center_ids_per_point = np.concatenate(all_center_ids).astype(np.int32)

#     return pts, ids, center_ids_per_point


# # =========================================================
# # 14. generate tactile touch points
# # =========================================================
# def generate_tactile_touch_points(
#     mesh,
#     surface_sample_n=200000,
#     num_rays=20000,
#     beam_radius=0.1,
#     rounds=5,
#     start_ratio=0.12,
#     end_ratio=0.03,
#     thickness_ratio=0.01,
#     points_per_finger=3000,
#     num_fingers=5,
# ):
#     spc = sample_from_mesh(mesh, n=surface_sample_n)
#     diag = bbox_diag(spc.points)

#     candidate_ids, hit_points = raycast_outer_hits(mesh, spc.points, num_rays=num_rays)

#     reachable_mask = build_global_reachable_mask(
#         spc.points,
#         hit_points,
#         beam_radius=beam_radius
#     )

#     scene = build_raycast_scene(mesh)

#     largest_radius = start_ratio * diag
#     smallest_radius = end_ratio * diag
#     thickness = thickness_ratio * diag

#     center_ids = sample_valid_fixed_centers(
#         scene=scene,
#         points=spc.points,
#         normals=spc.normals,
#         candidate_ids=candidate_ids,
#         reachable_mask=reachable_mask,
#         largest_radius=largest_radius,
#         thickness=thickness,
#         min_center_dist=2 * smallest_radius,
#         num_centers=num_fingers
#     )

#     print("[INFO] fixed center ids:", center_ids.tolist())

#     radius_schedule = np.linspace(start_ratio, end_ratio, rounds)

#     round_points = []
#     round_ids = []
#     finger_ids = []
#     center_ids_per_point = []

#     for rid, ratio in enumerate(radius_schedule):
#         radius = ratio * diag

#         print(f"\n[ROUND {rid}] radius_ratio={ratio:.4f}, radius={radius:.6f}")

#         pts, fids, cids = tactile_sampling_round(
#             spc=spc,
#             scene=scene,
#             reachable_mask=reachable_mask,
#             center_ids=center_ids,
#             radius=radius,
#             thickness=thickness,
#             points_per_finger=points_per_finger
#         )

#         if len(pts) == 0:
#             continue

#         round_points.append(pts)
#         round_ids.append(np.full(len(pts), rid, dtype=np.int32))
#         finger_ids.append(fids)
#         center_ids_per_point.append(cids)

#     if len(round_points) == 0:
#         raise RuntimeError("No tactile points were generated.")

#     touch_points = np.vstack(round_points).astype(np.float32)
#     touch_round_ids = np.concatenate(round_ids).astype(np.int32)
#     touch_finger_ids = np.concatenate(finger_ids).astype(np.int32)
#     touch_center_ids = np.concatenate(center_ids_per_point).astype(np.int32)
#     touch_centers = spc.points[center_ids].astype(np.float32)

#     return {
#         "touch_points": touch_points,
#         "touch_round_ids": touch_round_ids,
#         "touch_finger_ids": touch_finger_ids,
#         "touch_center_ids": touch_center_ids,
#         "touch_centers": touch_centers,
#     }


# # =========================================================
# # 15. Chou-style + uniform-space sampling
# # =========================================================
# def sample_training_points_chou_style(
#     mesh,
#     num_surface_points=235000,
#     sigma_large=0.005,
#     sigma_small=0.0005,
#     num_uniform_points=50000,
#     uniform_range=1.0,
# ):
#     print("[INFO] sampling surface/query points in Chou-style + uniform-space")

#     # 1) surface points
#     surface_points, face_idx = mesh.sample(num_surface_points, return_index=True)
#     surface_points = surface_points.astype(np.float32)

#     surface_normals = mesh.face_normals[face_idx].astype(np.float32)
#     surface_normals /= np.clip(
#         np.linalg.norm(surface_normals, axis=1, keepdims=True),
#         1e-8,
#         None
#     )

#     # 2) two near-surface Gaussian point sets
#     noise_large = np.random.normal(
#         loc=0.0,
#         scale=sigma_large,
#         size=surface_points.shape
#     ).astype(np.float32)

#     noise_small = np.random.normal(
#         loc=0.0,
#         scale=sigma_small,
#         size=surface_points.shape
#     ).astype(np.float32)

#     query_points_large = surface_points + noise_large
#     query_points_small = surface_points + noise_small

#     # 3) uniform space points
#     uniform_points = np.random.uniform(
#         low=-uniform_range,
#         high=uniform_range,
#         size=(num_uniform_points, 3)
#     ).astype(np.float32)

#     # 4) compute SDF separately
#     print("[INFO] computing near-surface sdf (large sigma) ...")
#     query_sdf_large = mesh_to_sdf(mesh, query_points_large).astype(np.float32)

#     print("[INFO] computing near-surface sdf (small sigma) ...")
#     query_sdf_small = mesh_to_sdf(mesh, query_points_small).astype(np.float32)

#     print("[INFO] computing uniform-space sdf ...")
#     uniform_sdf = mesh_to_sdf(mesh, uniform_points).astype(np.float32)

#     # 5) concatenate all query sets
#     query_points = np.concatenate(
#         [query_points_large, query_points_small, uniform_points],
#         axis=0
#     ).astype(np.float32)

#     query_sdf = np.concatenate(
#         [query_sdf_large, query_sdf_small, uniform_sdf],
#         axis=0
#     ).astype(np.float32)

#     surface_sdf = np.zeros((num_surface_points,), dtype=np.float32)

#     return {
#         "surface_points": surface_points,
#         "surface_normals": surface_normals,
#         "surface_sdf": surface_sdf,
#         "query_points": query_points,
#         "query_sdf": query_sdf,
#     }


# # =========================================================
# # 16. process single obj -> npz
# # =========================================================
# def process_single_obj_to_npz(
#     obj_path,
#     out_path,
#     tactile_surface_sample_n=200000,
#     tactile_num_rays=20000,
#     tactile_beam_radius=0.1,
#     tactile_rounds=5,
#     tactile_start_ratio=0.12,
#     tactile_end_ratio=0.03,
#     tactile_thickness_ratio=0.01,
#     tactile_points_per_finger=3000,
#     tactile_num_fingers=5,
#     num_surface_points=235000,
#     sigma_large=0.005,
#     sigma_small=0.0005,
#     num_uniform_points=50000,
# ):
#     print("\n==================================================")
#     print("[PROCESS]", obj_path)
#     print("==================================================")

#     mesh = trimesh.load(obj_path, force="mesh")
#     mesh.process(validate=True)
#     mesh = scale_to_unit_sphere(mesh)

#     tactile_data = generate_tactile_touch_points(
#         mesh=mesh,
#         surface_sample_n=tactile_surface_sample_n,
#         num_rays=tactile_num_rays,
#         beam_radius=tactile_beam_radius,
#         rounds=tactile_rounds,
#         start_ratio=tactile_start_ratio,
#         end_ratio=tactile_end_ratio,
#         thickness_ratio=tactile_thickness_ratio,
#         points_per_finger=tactile_points_per_finger,
#         num_fingers=tactile_num_fingers,
#     )

#     shape_data = sample_training_points_chou_style(
#         mesh=mesh,
#         num_surface_points=num_surface_points,
#         sigma_large=sigma_large,
#         sigma_small=sigma_small,
#         num_uniform_points=num_uniform_points,
#         uniform_range=1.0,
#     )

#     os.makedirs(os.path.dirname(out_path), exist_ok=True)

#     np.savez_compressed(
#         out_path,
#         surface_points=shape_data["surface_points"],
#         surface_normals=shape_data["surface_normals"],
#         surface_sdf=shape_data["surface_sdf"],
#         touch_points=tactile_data["touch_points"],
#         touch_round_ids=tactile_data["touch_round_ids"],
#         touch_finger_ids=tactile_data["touch_finger_ids"],
#         touch_center_ids=tactile_data["touch_center_ids"],
#         touch_centers=tactile_data["touch_centers"],
#         query_points=shape_data["query_points"],
#         query_sdf=shape_data["query_sdf"],
#         mesh_name=np.array(os.path.basename(obj_path)),
#     )

#     print("[SAVED]", out_path)
#     print("surface_points:", shape_data["surface_points"].shape)
#     print("touch_points  :", tactile_data["touch_points"].shape)
#     print("query_points  :", shape_data["query_points"].shape)


# # =========================================================
# # 17. process one split of one category
# # =========================================================
# def process_split(
#     category_dir,
#     split="train",
#     max_objects=80,
#     output_folder_name=None,
# ):
#     obj_dir = os.path.join(category_dir, f"{split}_obj")

#     if output_folder_name is None:
#         output_folder_name = f"tactistruct_npz_{split}"

#     out_dir = os.path.join(category_dir, output_folder_name)

#     if not os.path.isdir(obj_dir):
#         print(f"[WARN] skip missing dir: {obj_dir}")
#         return

#     os.makedirs(out_dir, exist_ok=True)

#     obj_files = sorted([
#         f for f in os.listdir(obj_dir)
#         if f.lower().endswith(".obj")
#     ])[:max_objects]

#     print(f"[INFO] {split}: found {len(obj_files)} obj files in {obj_dir}")

#     for name in obj_files:
#         obj_path = os.path.join(obj_dir, name)
#         out_path = os.path.join(out_dir, name[:-4] + ".npz")

#         if os.path.exists(out_path):
#             print("[SKIP exists]", out_path)
#             continue

#         try:
#             process_single_obj_to_npz(
#                 obj_path=obj_path,
#                 out_path=out_path,
#             )
#         except Exception as e:
#             print("[FAILED]", obj_path)
#             print("Error:", e)


# # =========================================================
# # 18. process all categories
# # =========================================================
# def process_all_categories(
#     root_dir,
#     split="train",
#     max_objects_per_category=80,
#     category_names=None,
# ):
#     subdirs = sorted([
#         os.path.join(root_dir, d)
#         for d in os.listdir(root_dir)
#         if os.path.isdir(os.path.join(root_dir, d))
#     ])

#     if category_names is not None:
#         category_names = set(category_names)
#         subdirs = [d for d in subdirs if os.path.basename(d) in category_names]

#     if not subdirs:
#         print("[WARN] no category folders found under:", root_dir)
#         return

#     print(f"[INFO] found {len(subdirs)} category folders under root.")

#     for category_dir in subdirs:
#         category_name = os.path.basename(category_dir)
#         print(f"\n########## Processing category: {category_name} ##########")

#         process_split(
#             category_dir=category_dir,
#             split=split,
#             max_objects=max_objects_per_category,
#             output_folder_name=f"tactistruct_npz_{split}",
#         )


# # =========================================================
# # 19. main
# # =========================================================
# if __name__ == "__main__":
#     root_dir = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40"

#     # 先单类别小规模测试更稳
#     process_all_categories(
#         root_dir=root_dir,
#         split="train",
#         max_objects_per_category=2,
#         category_names=["airplane"]
#     )

#     print("\nAll done.")



















# import os
# import numpy as np
# import trimesh
# import open3d as o3d

# from scipy.spatial import cKDTree
# from mesh_to_sdf import mesh_to_sdf


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
# # 3. dense surface sampling for tactile generation
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
# # 4. bbox diagonal
# # =========================================================
# def bbox_diag(points):
#     mn = points.min(axis=0)
#     mx = points.max(axis=0)
#     return float(np.linalg.norm(mx - mn))


# # =========================================================
# # 5. build raycasting scene
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
# # 6. ray casting (global outer-hit candidates)
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
# # 7. reachable region (global)
# # =========================================================
# def build_global_reachable_mask(points, hit_points, beam_radius):
#     print("[INFO] building reachable region")

#     hit_tree = cKDTree(hit_points)
#     dist, _ = hit_tree.query(points, k=1, distance_upper_bound=beam_radius)
#     reachable_mask = np.isfinite(dist)

#     print("[INFO] reachable points:", int(reachable_mask.sum()))
#     return reachable_mask


# # =========================================================
# # 8. tangent patch
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
# # 9. local visibility filter
# # =========================================================
# def filter_visible_points(scene,
#                           center,
#                           center_normal,
#                           candidate_points,
#                           eps=2e-3,
#                           hit_tol=2e-3):
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

#     ok = np.zeros(len(dirs), dtype=bool)
#     ok[finite_hit] = np.abs(t_hit[finite_hit] - dist[valid_dir][finite_hit]) <= hit_tol
#     visible[valid_indices] = ok

#     return visible


# # =========================================================
# # 10. patch + reachable + local visibility
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
#         points, normals, center_idx, radius, thickness,
#         normal_angle_deg, min_patch_points
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
# # 11. edge-aware soft contact probability
# # =========================================================
# def estimate_edge_deformation_factor(points,
#                                      normals,
#                                      center_idx,
#                                      radius,
#                                      edge_neighbor_ratio=0.35):
#     c = points[center_idx]
#     n0 = normals[center_idx]

#     d = np.linalg.norm(points - c[None, :], axis=1)
#     local_mask = d <= (edge_neighbor_ratio * radius)
#     local_ids = np.where(local_mask)[0]

#     if len(local_ids) < 20:
#         return 0.0

#     align = normals[local_ids] @ n0
#     align = np.clip(align, -1.0, 1.0)

#     normal_variation = 1.0 - np.mean(np.abs(align))
#     edge_factor = np.clip(normal_variation / 0.5, 0.0, 1.0)
#     return float(edge_factor)


# def compute_soft_contact_probability(points,
#                                      normals,
#                                      center_idx,
#                                      candidate_ids,
#                                      radius,
#                                      cross_surface_gain=0.35,
#                                      edge_neighbor_ratio=0.35):
#     if len(candidate_ids) == 0:
#         return np.zeros((0,), dtype=np.float64)

#     c = points[center_idx]
#     n0 = normals[center_idx]

#     pts = points[candidate_ids]
#     nrm = normals[candidate_ids]

#     v = pts - c[None, :]
#     dist = np.linalg.norm(v, axis=1)

#     radial = 1.0 - dist / max(radius, 1e-8)
#     radial = np.clip(radial, 0.0, None) ** 2

#     align = nrm @ n0
#     align_pos = np.clip(align, 0.0, 1.0)

#     edge_factor = estimate_edge_deformation_factor(
#         points=points,
#         normals=normals,
#         center_idx=center_idx,
#         radius=radius,
#         edge_neighbor_ratio=edge_neighbor_ratio
#     )

#     soft_cross = np.clip((align + 1.0) * 0.5, 0.0, 1.0) ** 2

#     prob = radial * (align_pos + cross_surface_gain * edge_factor * soft_cross)

#     if np.all(prob <= 1e-12):
#         prob = np.ones_like(prob, dtype=np.float64)

#     prob = prob.astype(np.float64)
#     prob /= prob.sum()
#     return prob


# # =========================================================
# # 12. center sampling
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
# # 13. tactile sampling round
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
#     all_center_ids = []

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
#             continue

#         idx = np.where(mask)[0]

#         prob = compute_soft_contact_probability(
#             points=points,
#             normals=normals,
#             center_idx=center_idx,
#             candidate_ids=idx,
#             radius=radius,
#             cross_surface_gain=0.35,
#             edge_neighbor_ratio=0.35
#         )

#         choose = rng.choice(
#             idx,
#             points_per_finger,
#             replace=len(idx) < points_per_finger,
#             p=prob
#         )

#         all_pts.append(points[choose])
#         all_ids.append(np.full(points_per_finger, fid, dtype=np.int32))
#         all_center_ids.append(np.full(points_per_finger, center_idx, dtype=np.int32))

#     if len(all_pts) == 0:
#         return (
#             np.zeros((0, 3), dtype=np.float32),
#             np.zeros((0,), dtype=np.int32),
#             np.zeros((0,), dtype=np.int32),
#         )

#     pts = np.vstack(all_pts).astype(np.float32)
#     ids = np.concatenate(all_ids).astype(np.int32)
#     center_ids_per_point = np.concatenate(all_center_ids).astype(np.int32)

#     return pts, ids, center_ids_per_point


# # =========================================================
# # 14. generate one tactile observation
# # =========================================================
# def generate_tactile_touch_points(
#     mesh,
#     surface_sample_n=200000,
#     num_rays=20000,
#     beam_radius=0.1,
#     rounds=5,
#     start_ratio=0.12,
#     end_ratio=0.03,
#     thickness_ratio=0.01,
#     points_per_finger=3000,
#     num_fingers=5,
# ):
#     spc = sample_from_mesh(mesh, n=surface_sample_n)
#     diag = bbox_diag(spc.points)

#     candidate_ids, hit_points = raycast_outer_hits(mesh, spc.points, num_rays=num_rays)

#     reachable_mask = build_global_reachable_mask(
#         spc.points,
#         hit_points,
#         beam_radius=beam_radius
#     )

#     scene = build_raycast_scene(mesh)

#     largest_radius = start_ratio * diag
#     smallest_radius = end_ratio * diag
#     thickness = thickness_ratio * diag

#     center_ids = sample_valid_fixed_centers(
#         scene=scene,
#         points=spc.points,
#         normals=spc.normals,
#         candidate_ids=candidate_ids,
#         reachable_mask=reachable_mask,
#         largest_radius=largest_radius,
#         thickness=thickness,
#         min_center_dist=2 * smallest_radius,
#         num_centers=num_fingers
#     )

#     radius_schedule = np.linspace(start_ratio, end_ratio, rounds)

#     round_points = []
#     round_ids = []
#     finger_ids = []
#     center_ids_per_point = []

#     for rid, ratio in enumerate(radius_schedule):
#         radius = ratio * diag

#         pts, fids, cids = tactile_sampling_round(
#             spc=spc,
#             scene=scene,
#             reachable_mask=reachable_mask,
#             center_ids=center_ids,
#             radius=radius,
#             thickness=thickness,
#             points_per_finger=points_per_finger
#         )

#         if len(pts) == 0:
#             continue

#         round_points.append(pts)
#         round_ids.append(np.full(len(pts), rid, dtype=np.int32))
#         finger_ids.append(fids)
#         center_ids_per_point.append(cids)

#     if len(round_points) == 0:
#         raise RuntimeError("No tactile points were generated.")

#     touch_points = np.vstack(round_points).astype(np.float32)
#     touch_round_ids = np.concatenate(round_ids).astype(np.int32)
#     touch_finger_ids = np.concatenate(finger_ids).astype(np.int32)
#     touch_center_ids = np.concatenate(center_ids_per_point).astype(np.int32)
#     touch_centers = spc.points[center_ids].astype(np.float32)

#     return {
#         "touch_points": touch_points,
#         "touch_round_ids": touch_round_ids,
#         "touch_finger_ids": touch_finger_ids,
#         "touch_center_ids": touch_center_ids,
#         "touch_centers": touch_centers,
#     }


# # =========================================================
# # 15. shape/query sampling, computed once per mesh
# # =========================================================
# def sample_training_points_chou_style(
#     mesh,
#     num_surface_points=235000,
#     sigma_large=0.005,
#     sigma_small=0.0005,
#     num_uniform_points=50000,
#     uniform_range=1.0,
# ):
#     print("[INFO] sampling shape/query points once")

#     surface_points, face_idx = mesh.sample(num_surface_points, return_index=True)
#     surface_points = surface_points.astype(np.float32)

#     surface_normals = mesh.face_normals[face_idx].astype(np.float32)
#     surface_normals /= np.clip(
#         np.linalg.norm(surface_normals, axis=1, keepdims=True),
#         1e-8,
#         None
#     )

#     noise_large = np.random.normal(0.0, sigma_large, size=surface_points.shape).astype(np.float32)
#     noise_small = np.random.normal(0.0, sigma_small, size=surface_points.shape).astype(np.float32)

#     query_points_large = surface_points + noise_large
#     query_points_small = surface_points + noise_small

#     uniform_points = np.random.uniform(
#         low=-uniform_range,
#         high=uniform_range,
#         size=(num_uniform_points, 3)
#     ).astype(np.float32)

#     print("[INFO] computing sdf for query points ...")
#     query_sdf_large = mesh_to_sdf(mesh, query_points_large).astype(np.float32)
#     query_sdf_small = mesh_to_sdf(mesh, query_points_small).astype(np.float32)
#     uniform_sdf = mesh_to_sdf(mesh, uniform_points).astype(np.float32)

#     query_points = np.concatenate(
#         [query_points_large, query_points_small, uniform_points],
#         axis=0
#     ).astype(np.float32)

#     query_sdf = np.concatenate(
#         [query_sdf_large, query_sdf_small, uniform_sdf],
#         axis=0
#     ).astype(np.float32)

#     surface_sdf = np.zeros((num_surface_points,), dtype=np.float32)

#     return {
#         "surface_points": surface_points,
#         "surface_normals": surface_normals,
#         "surface_sdf": surface_sdf,
#         "query_points": query_points,
#         "query_sdf": query_sdf,
#     }


# # =========================================================
# # 16. one mesh -> multiple tactile npz
# # =========================================================
# def process_single_obj_to_multiple_npz(
#     obj_path,
#     out_dir,
#     num_tactile_augs=10,
#     tactile_surface_sample_n=200000,
#     tactile_num_rays=20000,
#     tactile_beam_radius=0.1,
#     tactile_rounds=5,
#     tactile_start_ratio=0.12,
#     tactile_end_ratio=0.03,
#     tactile_thickness_ratio=0.01,
#     tactile_points_per_finger=3000,
#     tactile_num_fingers=5,
#     num_surface_points=235000,
#     sigma_large=0.005,
#     sigma_small=0.0005,
#     num_uniform_points=50000,
# ):
#     print("\n==================================================")
#     print("[PROCESS]", obj_path)
#     print("==================================================")

#     mesh_name = os.path.splitext(os.path.basename(obj_path))[0]

#     mesh = trimesh.load(obj_path, force="mesh")
#     mesh.process(validate=True)
#     mesh = scale_to_unit_sphere(mesh)

#     # shape/query part only once
#     shape_data = sample_training_points_chou_style(
#         mesh=mesh,
#         num_surface_points=num_surface_points,
#         sigma_large=sigma_large,
#         sigma_small=sigma_small,
#         num_uniform_points=num_uniform_points,
#         uniform_range=1.0,
#     )

#     os.makedirs(out_dir, exist_ok=True)

#     for aug_idx in range(num_tactile_augs):
#         print(f"\n---------- tactile augmentation {aug_idx+1}/{num_tactile_augs} ----------")

#         tactile_data = generate_tactile_touch_points(
#             mesh=mesh,
#             surface_sample_n=tactile_surface_sample_n,
#             num_rays=tactile_num_rays,
#             beam_radius=tactile_beam_radius,
#             rounds=tactile_rounds,
#             start_ratio=tactile_start_ratio,
#             end_ratio=tactile_end_ratio,
#             thickness_ratio=tactile_thickness_ratio,
#             points_per_finger=tactile_points_per_finger,
#             num_fingers=tactile_num_fingers,
#         )

#         out_path = os.path.join(out_dir, f"{mesh_name}_aug{aug_idx:02d}.npz")

#         np.savez_compressed(
#             out_path,
#             surface_points=shape_data["surface_points"],
#             surface_normals=shape_data["surface_normals"],
#             surface_sdf=shape_data["surface_sdf"],
#             touch_points=tactile_data["touch_points"],
#             touch_round_ids=tactile_data["touch_round_ids"],
#             touch_finger_ids=tactile_data["touch_finger_ids"],
#             touch_center_ids=tactile_data["touch_center_ids"],
#             touch_centers=tactile_data["touch_centers"],
#             query_points=shape_data["query_points"],
#             query_sdf=shape_data["query_sdf"],
#             mesh_name=np.array(mesh_name),
#             aug_index=np.array(aug_idx, dtype=np.int32),
#         )

#         print("[SAVED]", out_path)
#         print("surface_points:", shape_data["surface_points"].shape)
#         print("touch_points  :", tactile_data["touch_points"].shape)
#         print("query_points  :", shape_data["query_points"].shape)


# # =========================================================
# # 17. process one split
# # =========================================================
# def process_split(
#     category_dir,
#     split="train",
#     max_objects=80,
#     num_tactile_augs=10,
#     output_folder_name=None,
# ):
#     obj_dir = os.path.join(category_dir, f"{split}_obj")

#     if output_folder_name is None:
#         output_folder_name = f"tactistruct_npz_{split}"

#     out_dir = os.path.join(category_dir, output_folder_name)

#     if not os.path.isdir(obj_dir):
#         print(f"[WARN] skip missing dir: {obj_dir}")
#         return

#     os.makedirs(out_dir, exist_ok=True)

#     obj_files = sorted([
#         f for f in os.listdir(obj_dir)
#         if f.lower().endswith(".obj")
#     ])[:max_objects]

#     print(f"[INFO] {split}: found {len(obj_files)} obj files in {obj_dir}")

#     for name in obj_files:
#         obj_path = os.path.join(obj_dir, name)
#         mesh_name = os.path.splitext(name)[0]

#         # 如果 10 个都存在，就跳过
#         expected_paths = [
#             os.path.join(out_dir, f"{mesh_name}_aug{i:02d}.npz")
#             for i in range(num_tactile_augs)
#         ]
#         if all(os.path.exists(p) for p in expected_paths):
#             print(f"[SKIP exists] all augmentations already exist for {mesh_name}")
#             continue

#         try:
#             process_single_obj_to_multiple_npz(
#                 obj_path=obj_path,
#                 out_dir=out_dir,
#                 num_tactile_augs=num_tactile_augs,
#             )
#         except Exception as e:
#             print("[FAILED]", obj_path)
#             print("Error:", e)


# # =========================================================
# # 18. process all categories
# # =========================================================
# def process_all_categories(
#     root_dir,
#     split="train",
#     max_objects_per_category=80,
#     num_tactile_augs=10,
#     category_names=None,
# ):
#     subdirs = sorted([
#         os.path.join(root_dir, d)
#         for d in os.listdir(root_dir)
#         if os.path.isdir(os.path.join(root_dir, d))
#     ])

#     if category_names is not None:
#         category_names = set(category_names)
#         subdirs = [d for d in subdirs if os.path.basename(d) in category_names]

#     if not subdirs:
#         print("[WARN] no category folders found under:", root_dir)
#         return

#     print(f"[INFO] found {len(subdirs)} category folders under root.")

#     for category_dir in subdirs:
#         category_name = os.path.basename(category_dir)
#         print(f"\n########## Processing category: {category_name} ##########")

#         process_split(
#             category_dir=category_dir,
#             split=split,
#             max_objects=max_objects_per_category,
#             num_tactile_augs=num_tactile_augs,
#             output_folder_name=f"tactistruct_npz_{split}",
#         )


# # =========================================================
# # 19. main
# # =========================================================
# if __name__ == "__main__":
#     root_dir = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40"

#     # 先用一个类别测试
#     process_all_categories(
#         root_dir=root_dir,
#         split="train",
#         max_objects_per_category=2,
#         num_tactile_augs=10,
#         category_names=["airplane"]
#     )

#     print("\nAll done.")




















# import os
# import numpy as np
# import trimesh
# import open3d as o3d

# from scipy.spatial import cKDTree
# from mesh_to_sdf import sample_sdf_near_surface


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
# # 3. dense surface sampling for tactile generation
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
# # 4. explicit surface points for storage
# # =========================================================
# def sample_surface_points_for_storage(mesh, num_surface_points=235000):
#     surface_points, face_idx = mesh.sample(num_surface_points, return_index=True)
#     surface_points = surface_points.astype(np.float32)

#     surface_normals = mesh.face_normals[face_idx].astype(np.float32)
#     surface_normals /= np.clip(
#         np.linalg.norm(surface_normals, axis=1, keepdims=True),
#         1e-8,
#         None
#     )

#     return surface_points, surface_normals


# # =========================================================
# # 5. bbox diagonal
# # =========================================================
# def bbox_diag(points):
#     mn = points.min(axis=0)
#     mx = points.max(axis=0)
#     return float(np.linalg.norm(mx - mn))


# # =========================================================
# # 6. build raycasting scene
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
# # 7. ray casting (global outer-hit candidates)
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
# # 8. reachable region (global)
# # =========================================================
# def build_global_reachable_mask(points, hit_points, beam_radius):
#     print("[INFO] building reachable region")

#     hit_tree = cKDTree(hit_points)
#     dist, _ = hit_tree.query(points, k=1, distance_upper_bound=beam_radius)
#     reachable_mask = np.isfinite(dist)

#     print("[INFO] reachable points:", int(reachable_mask.sum()))
#     return reachable_mask


# # =========================================================
# # 9. tangent patch
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
# # 10. local visibility filter
# # =========================================================
# def filter_visible_points(scene,
#                           center,
#                           center_normal,
#                           candidate_points,
#                           eps=2e-3,
#                           hit_tol=2e-3):
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

#     ok = np.zeros(len(dirs), dtype=bool)
#     ok[finite_hit] = np.abs(t_hit[finite_hit] - dist[valid_dir][finite_hit]) <= hit_tol
#     visible[valid_indices] = ok

#     return visible


# # =========================================================
# # 11. patch + reachable + visibility
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
#         points, normals, center_idx, radius, thickness,
#         normal_angle_deg, min_patch_points
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
# # 12. edge-aware soft contact probability
# # =========================================================
# def estimate_edge_deformation_factor(points,
#                                      normals,
#                                      center_idx,
#                                      radius,
#                                      edge_neighbor_ratio=0.35):
#     c = points[center_idx]
#     n0 = normals[center_idx]

#     d = np.linalg.norm(points - c[None, :], axis=1)
#     local_mask = d <= (edge_neighbor_ratio * radius)
#     local_ids = np.where(local_mask)[0]

#     if len(local_ids) < 20:
#         return 0.0

#     align = normals[local_ids] @ n0
#     align = np.clip(align, -1.0, 1.0)

#     normal_variation = 1.0 - np.mean(np.abs(align))
#     edge_factor = np.clip(normal_variation / 0.5, 0.0, 1.0)
#     return float(edge_factor)


# def compute_soft_contact_probability(points,
#                                      normals,
#                                      center_idx,
#                                      candidate_ids,
#                                      radius,
#                                      cross_surface_gain=0.35,
#                                      edge_neighbor_ratio=0.35):
#     if len(candidate_ids) == 0:
#         return np.zeros((0,), dtype=np.float64)

#     c = points[center_idx]
#     n0 = normals[center_idx]

#     pts = points[candidate_ids]
#     nrm = normals[candidate_ids]

#     v = pts - c[None, :]
#     dist = np.linalg.norm(v, axis=1)

#     radial = 1.0 - dist / max(radius, 1e-8)
#     radial = np.clip(radial, 0.0, None) ** 2

#     align = nrm @ n0
#     align_pos = np.clip(align, 0.0, 1.0)

#     edge_factor = estimate_edge_deformation_factor(
#         points=points,
#         normals=normals,
#         center_idx=center_idx,
#         radius=radius,
#         edge_neighbor_ratio=edge_neighbor_ratio
#     )

#     soft_cross = np.clip((align + 1.0) * 0.5, 0.0, 1.0) ** 2

#     prob = radial * (align_pos + cross_surface_gain * edge_factor * soft_cross)

#     if np.all(prob <= 1e-12):
#         prob = np.ones_like(prob, dtype=np.float64)

#     prob = prob.astype(np.float64)
#     prob /= prob.sum()
#     return prob


# # =========================================================
# # 13. center sampling
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

#     if len(centers) < num_centers:
#         raise RuntimeError(
#             f"Only found {len(centers)} valid centers, need {num_centers}. "
#             f"Try increasing sample count or relaxing thresholds."
#         )

#     return np.array(centers, dtype=np.int64)


# # =========================================================
# # 14. tactile sampling round
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
#     all_center_ids = []

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
#             continue

#         idx = np.where(mask)[0]

#         prob = compute_soft_contact_probability(
#             points=points,
#             normals=normals,
#             center_idx=center_idx,
#             candidate_ids=idx,
#             radius=radius,
#             cross_surface_gain=0.35,
#             edge_neighbor_ratio=0.35
#         )

#         choose = rng.choice(
#             idx,
#             points_per_finger,
#             replace=len(idx) < points_per_finger,
#             p=prob
#         )

#         all_pts.append(points[choose])
#         all_ids.append(np.full(points_per_finger, fid, dtype=np.int32))
#         all_center_ids.append(np.full(points_per_finger, center_idx, dtype=np.int32))

#     if len(all_pts) == 0:
#         return (
#             np.zeros((0, 3), dtype=np.float32),
#             np.zeros((0,), dtype=np.int32),
#             np.zeros((0,), dtype=np.int32),
#         )

#     pts = np.vstack(all_pts).astype(np.float32)
#     ids = np.concatenate(all_ids).astype(np.int32)
#     center_ids_per_point = np.concatenate(all_center_ids).astype(np.int32)

#     return pts, ids, center_ids_per_point


# # =========================================================
# # 15. generate one tactile observation
# # =========================================================
# def generate_tactile_touch_points(
#     mesh,
#     surface_sample_n=200000,
#     num_rays=20000,
#     beam_radius=0.1,
#     rounds=5,
#     start_ratio=0.12,
#     end_ratio=0.03,
#     thickness_ratio=0.01,
#     points_per_finger=3000,
#     num_fingers=5,
# ):
#     spc = sample_from_mesh(mesh, n=surface_sample_n)
#     diag = bbox_diag(spc.points)

#     candidate_ids, hit_points = raycast_outer_hits(mesh, spc.points, num_rays=num_rays)
#     reachable_mask = build_global_reachable_mask(spc.points, hit_points, beam_radius=beam_radius)
#     scene = build_raycast_scene(mesh)

#     largest_radius = start_ratio * diag
#     smallest_radius = end_ratio * diag
#     thickness = thickness_ratio * diag

#     center_ids = sample_valid_fixed_centers(
#         scene=scene,
#         points=spc.points,
#         normals=spc.normals,
#         candidate_ids=candidate_ids,
#         reachable_mask=reachable_mask,
#         largest_radius=largest_radius,
#         thickness=thickness,
#         min_center_dist=2 * smallest_radius,
#         num_centers=num_fingers
#     )

#     radius_schedule = np.linspace(start_ratio, end_ratio, rounds)

#     round_points = []
#     round_ids = []
#     finger_ids = []
#     center_ids_per_point = []

#     for rid, ratio in enumerate(radius_schedule):
#         radius = ratio * diag

#         pts, fids, cids = tactile_sampling_round(
#             spc=spc,
#             scene=scene,
#             reachable_mask=reachable_mask,
#             center_ids=center_ids,
#             radius=radius,
#             thickness=thickness,
#             points_per_finger=points_per_finger
#         )

#         if len(pts) == 0:
#             continue

#         round_points.append(pts)
#         round_ids.append(np.full(len(pts), rid, dtype=np.int32))
#         finger_ids.append(fids)
#         center_ids_per_point.append(cids)

#     if len(round_points) == 0:
#         raise RuntimeError("No tactile points were generated.")

#     touch_points = np.vstack(round_points).astype(np.float32)
#     touch_round_ids = np.concatenate(round_ids).astype(np.int32)
#     touch_finger_ids = np.concatenate(finger_ids).astype(np.int32)
#     touch_center_ids = np.concatenate(center_ids_per_point).astype(np.int32)
#     touch_centers = spc.points[center_ids].astype(np.float32)

#     return {
#         "touch_points": touch_points,
#         "touch_round_ids": touch_round_ids,
#         "touch_finger_ids": touch_finger_ids,
#         "touch_center_ids": touch_center_ids,
#         "touch_centers": touch_centers,
#     }


# # =========================================================
# # 16. one mesh -> one npz with 10 tactile groups
# # =========================================================
# def process_single_obj_to_merged_npz(
#     obj_path,
#     out_path,
#     num_tactile_samples=10,
#     tactile_surface_sample_n=200000,
#     tactile_num_rays=20000,
#     tactile_beam_radius=0.1,
#     tactile_rounds=5,
#     tactile_start_ratio=0.12,
#     tactile_end_ratio=0.03,
#     tactile_thickness_ratio=0.01,
#     tactile_points_per_finger=3000,
#     tactile_num_fingers=5,
#     num_surface_points=235000,
#     num_query_points=250000,
# ):
#     print("\n==================================================")
#     print("[PROCESS]", obj_path)
#     print("==================================================")

#     mesh_name = os.path.splitext(os.path.basename(obj_path))[0]

#     mesh = trimesh.load(obj_path, force="mesh")
#     mesh.process(validate=True)
#     mesh = scale_to_unit_sphere(mesh)

#     # -------------------------
#     # surface_points / normals
#     # -------------------------
#     surface_points, surface_normals = sample_surface_points_for_storage(
#         mesh,
#         num_surface_points=num_surface_points
#     )

#     # -------------------------
#     # query_points / query_sdf
#     # use sample_sdf_near_surface
#     # -------------------------
#     print("[INFO] sampling query points using sample_sdf_near_surface ...")
#     query_points, query_sdf = sample_sdf_near_surface(
#         mesh,
#         number_of_points=num_query_points
#     )
#     query_points = query_points.astype(np.float32)
#     query_sdf = query_sdf.astype(np.float32)

#     # -------------------------
#     # 10 tactile observations
#     # -------------------------
#     touch_points_all = []
#     touch_round_ids_all = []
#     touch_finger_ids_all = []
#     touch_center_ids_all = []
#     touch_centers_all = []

#     for sample_idx in range(num_tactile_samples):
#         print(f"\n---------- tactile sample {sample_idx + 1}/{num_tactile_samples} ----------")

#         tactile_data = generate_tactile_touch_points(
#             mesh=mesh,
#             surface_sample_n=tactile_surface_sample_n,
#             num_rays=tactile_num_rays,
#             beam_radius=tactile_beam_radius,
#             rounds=tactile_rounds,
#             start_ratio=tactile_start_ratio,
#             end_ratio=tactile_end_ratio,
#             thickness_ratio=tactile_thickness_ratio,
#             points_per_finger=tactile_points_per_finger,
#             num_fingers=tactile_num_fingers,
#         )

#         touch_points_all.append(tactile_data["touch_points"])
#         touch_round_ids_all.append(tactile_data["touch_round_ids"])
#         touch_finger_ids_all.append(tactile_data["touch_finger_ids"])
#         touch_center_ids_all.append(tactile_data["touch_center_ids"])
#         touch_centers_all.append(tactile_data["touch_centers"])

#     touch_points_all = np.stack(touch_points_all, axis=0).astype(np.float32)
#     touch_round_ids_all = np.stack(touch_round_ids_all, axis=0).astype(np.int32)
#     touch_finger_ids_all = np.stack(touch_finger_ids_all, axis=0).astype(np.int32)
#     touch_center_ids_all = np.stack(touch_center_ids_all, axis=0).astype(np.int32)
#     touch_centers_all = np.stack(touch_centers_all, axis=0).astype(np.float32)

#     os.makedirs(os.path.dirname(out_path), exist_ok=True)

#     np.savez_compressed(
#         out_path,
#         surface_points=surface_points,
#         surface_normals=surface_normals,
#         touch_points=touch_points_all,
#         touch_round_ids=touch_round_ids_all,
#         touch_finger_ids=touch_finger_ids_all,
#         touch_center_ids=touch_center_ids_all,
#         touch_centers=touch_centers_all,
#         query_points=query_points,
#         query_sdf=query_sdf,
#         mesh_name=np.array(mesh_name),
#         num_tactile_samples=np.array(num_tactile_samples, dtype=np.int32),
#     )

#     print("[SAVED]", out_path)
#     print("surface_points   :", surface_points.shape)
#     print("surface_normals  :", surface_normals.shape)
#     print("query_points     :", query_points.shape)
#     print("query_sdf        :", query_sdf.shape)
#     print("touch_points     :", touch_points_all.shape)
#     print("touch_round_ids  :", touch_round_ids_all.shape)
#     print("touch_finger_ids :", touch_finger_ids_all.shape)
#     print("touch_center_ids :", touch_center_ids_all.shape)
#     print("touch_centers    :", touch_centers_all.shape)


# # =========================================================
# # 17. process one split
# # =========================================================
# def process_split(
#     category_dir,
#     split="train",
#     max_objects=80,
#     num_tactile_samples=10,
#     output_folder_name=None,
# ):
#     obj_dir = os.path.join(category_dir, f"{split}_obj")

#     if output_folder_name is None:
#         output_folder_name = f"tactistruct_npz_{split}"

#     out_dir = os.path.join(category_dir, output_folder_name)

#     if not os.path.isdir(obj_dir):
#         print(f"[WARN] skip missing dir: {obj_dir}")
#         return

#     os.makedirs(out_dir, exist_ok=True)

#     obj_files = sorted([
#         f for f in os.listdir(obj_dir)
#         if f.lower().endswith(".obj")
#     ])[:max_objects]

#     print(f"[INFO] {split}: found {len(obj_files)} obj files in {obj_dir}")

#     for name in obj_files:
#         obj_path = os.path.join(obj_dir, name)
#         out_path = os.path.join(out_dir, os.path.splitext(name)[0] + ".npz")

#         if os.path.exists(out_path):
#             print("[SKIP exists]", out_path)
#             continue

#         try:
#             process_single_obj_to_merged_npz(
#                 obj_path=obj_path,
#                 out_path=out_path,
#                 num_tactile_samples=num_tactile_samples,
#             )
#         except Exception as e:
#             print("[FAILED]", obj_path)
#             print("Error:", e)


# # =========================================================
# # 18. process all categories
# # =========================================================
# def process_all_categories(
#     root_dir,
#     split="train",
#     max_objects_per_category=80,
#     num_tactile_samples=10,
#     category_names=None,
# ):
#     subdirs = sorted([
#         os.path.join(root_dir, d)
#         for d in os.listdir(root_dir)
#         if os.path.isdir(os.path.join(root_dir, d))
#     ])

#     if category_names is not None:
#         category_names = set(category_names)
#         subdirs = [d for d in subdirs if os.path.basename(d) in category_names]

#     if not subdirs:
#         print("[WARN] no category folders found under:", root_dir)
#         return

#     print(f"[INFO] found {len(subdirs)} category folders under root.")

#     for category_dir in subdirs:
#         category_name = os.path.basename(category_dir)
#         print(f"\n########## Processing category: {category_name} ##########")

#         process_split(
#             category_dir=category_dir,
#             split=split,
#             max_objects=max_objects_per_category,
#             num_tactile_samples=num_tactile_samples,
#             output_folder_name=f"tactistruct_npz_{split}",
#         )


# # =========================================================
# # 19. main
# # =========================================================
# if __name__ == "__main__":
#     root_dir = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40"

#     process_all_categories(
#         root_dir=root_dir,
#         split="train",
#         max_objects_per_category=2,
#         num_tactile_samples=10,
#         category_names=["airplane"]
#     )

#     print("\nAll done.")














# import os
# import numpy as np
# import trimesh
# import open3d as o3d

# from scipy.spatial import cKDTree
# from mesh_to_sdf import sample_sdf_near_surface


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
# # 3. dense surface sampling
# # =========================================================
# def sample_from_mesh(mesh, n=200000):
#     points, face_idx = mesh.sample(n, return_index=True)

#     normals = mesh.face_normals[face_idx].astype(np.float32)
#     normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12, None)

#     return SurfacePointCloud(
#         mesh=mesh,
#         points=points.astype(np.float32),
#         normals=normals.astype(np.float32),
#     )


# # =========================================================
# # 4. explicit surface storage
# # =========================================================
# def sample_surface_points_for_storage(mesh, num_surface_points=235000):
#     surface_points, face_idx = mesh.sample(num_surface_points, return_index=True)
#     surface_points = surface_points.astype(np.float32)

#     surface_normals = mesh.face_normals[face_idx].astype(np.float32)
#     surface_normals /= np.clip(
#         np.linalg.norm(surface_normals, axis=1, keepdims=True),
#         1e-8,
#         None
#     )

#     return surface_points, surface_normals


# # =========================================================
# # 5. bbox diagonal
# # =========================================================
# def bbox_diag(points):
#     mn = points.min(axis=0)
#     mx = points.max(axis=0)
#     return float(np.linalg.norm(mx - mn))


# # =========================================================
# # 6. raycasting scene
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
# # 7. ray casting outer hits
# # =========================================================
# def raycast_outer_hits(mesh, points, num_rays=20000, rng=None):
#     if rng is None:
#         rng = np.random.default_rng()

#     scene = build_raycast_scene(mesh)

#     dirs = rng.normal(size=(num_rays, 3))
#     dirs /= np.clip(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12, None)

#     origins = dirs * 3.0
#     directions = -dirs

#     rays = np.concatenate([origins, directions], axis=1).astype(np.float32)
#     rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

#     ans = scene.cast_rays(rays)
#     t_hit = ans["t_hit"].numpy()

#     valid = np.isfinite(t_hit)
#     hit_points = origins[valid] + directions[valid] * t_hit[valid][:, None]

#     if len(hit_points) == 0:
#         raise RuntimeError("No outer ray hits found.")

#     tree = cKDTree(points)
#     _, ids = tree.query(hit_points, k=1)
#     candidate_ids = np.unique(ids.astype(np.int64))

#     return candidate_ids, hit_points.astype(np.float32)


# # =========================================================
# # 8. reachable region
# # =========================================================
# def build_global_reachable_mask(points, hit_points, beam_radius):
#     hit_tree = cKDTree(hit_points)
#     dist, _ = hit_tree.query(points, k=1, distance_upper_bound=beam_radius)
#     reachable_mask = np.isfinite(dist)
#     return reachable_mask


# # =========================================================
# # 9. theoretical tangent patch
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

#     v = points - c
#     height = v @ n
#     v_plane = v - height[:, None] * n
#     plane_dist = np.linalg.norm(v_plane, axis=1)

#     plane_mask = plane_dist <= radius
#     thickness_mask = np.abs(height) <= thickness

#     cos_th = np.cos(np.deg2rad(normal_angle_deg))
#     normal_mask = (normals @ n) >= cos_th

#     mask = plane_mask & thickness_mask & normal_mask

#     if int(mask.sum()) < int(min_patch_points):
#         return None

#     return mask


# # =========================================================
# # 10. local visibility
# # =========================================================
# def filter_visible_points(
#     scene,
#     center,
#     center_normal,
#     candidate_points,
#     eps=2e-3,
#     hit_tol=2e-3,
# ):
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

#     ok = np.zeros(len(dirs), dtype=bool)
#     ok[finite_hit] = np.abs(t_hit[finite_hit] - dist[valid_dir][finite_hit]) <= hit_tol
#     visible[valid_indices] = ok

#     return visible


# # =========================================================
# # 11. patch + reachable + visibility
# # =========================================================
# def extract_patch_with_visibility(
#     scene,
#     points,
#     normals,
#     reachable_mask,
#     center_idx,
#     radius,
#     thickness,
#     normal_angle_deg=28.0,
#     min_patch_points=150,
#     min_final_points=80,
#     min_reachable_coverage=0.2,
#     min_visible_coverage=0.3,
#     eps=1e-3,
#     hit_tol=3e-3,
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
#         return None, {
#             "reason": "theoretical_patch_too_small",
#             "coverage_reachable": 0.0,
#             "coverage_visible": 0.0,
#             "n_patch": 0,
#             "n_reachable": 0,
#             "n_visible": 0,
#             "candidate_ids": np.zeros((0,), dtype=np.int64),
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
#             "n_visible": 0,
#             "candidate_ids": reachable_ids.astype(np.int64),
#         }

#     if len(reachable_ids) < min_final_points:
#         return None, {
#             "reason": "reachable_points_too_few",
#             "coverage_reachable": coverage_reachable,
#             "coverage_visible": 0.0,
#             "n_patch": len(patch_ids),
#             "n_reachable": len(reachable_ids),
#             "n_visible": 0,
#             "candidate_ids": reachable_ids.astype(np.int64),
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
#         hit_tol=hit_tol,
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
#             "n_visible": len(visible_ids),
#             "candidate_ids": visible_ids.astype(np.int64),
#         }

#     if len(visible_ids) < min_final_points:
#         return None, {
#             "reason": "visible_points_too_few",
#             "coverage_reachable": coverage_reachable,
#             "coverage_visible": coverage_visible,
#             "n_patch": len(patch_ids),
#             "n_reachable": len(reachable_ids),
#             "n_visible": len(visible_ids),
#             "candidate_ids": visible_ids.astype(np.int64),
#         }

#     final_mask = np.zeros(len(points), dtype=bool)
#     final_mask[visible_ids] = True

#     return final_mask, {
#         "reason": "ok",
#         "coverage_reachable": coverage_reachable,
#         "coverage_visible": coverage_visible,
#         "n_patch": len(patch_ids),
#         "n_reachable": len(reachable_ids),
#         "n_visible": len(visible_ids),
#         "candidate_ids": visible_ids.astype(np.int64),
#     }


# # =========================================================
# # 12. edge-aware probability
# # =========================================================
# def estimate_edge_deformation_factor(
#     points,
#     normals,
#     center_idx,
#     radius,
#     edge_neighbor_ratio=0.35,
# ):
#     c = points[center_idx]
#     n0 = normals[center_idx]

#     d = np.linalg.norm(points - c[None, :], axis=1)
#     local_mask = d <= (edge_neighbor_ratio * radius)
#     local_ids = np.where(local_mask)[0]

#     if len(local_ids) < 20:
#         return 0.0

#     align = normals[local_ids] @ n0
#     align = np.clip(align, -1.0, 1.0)

#     normal_variation = 1.0 - np.mean(np.abs(align))
#     edge_factor = np.clip(normal_variation / 0.5, 0.0, 1.0)
#     return float(edge_factor)


# def compute_soft_contact_probability(
#     points,
#     normals,
#     center_idx,
#     candidate_ids,
#     radius,
#     cross_surface_gain=0.35,
#     edge_neighbor_ratio=0.35,
# ):
#     if len(candidate_ids) == 0:
#         return np.zeros((0,), dtype=np.float64)

#     c = points[center_idx]
#     n0 = normals[center_idx]

#     pts = points[candidate_ids]
#     nrm = normals[candidate_ids]

#     v = pts - c[None, :]
#     dist = np.linalg.norm(v, axis=1)

#     radial = 1.0 - dist / max(radius, 1e-8)
#     radial = np.clip(radial, 0.0, None) ** 2

#     align = nrm @ n0
#     align_pos = np.clip(align, 0.0, 1.0)

#     edge_factor = estimate_edge_deformation_factor(
#         points=points,
#         normals=normals,
#         center_idx=center_idx,
#         radius=radius,
#         edge_neighbor_ratio=edge_neighbor_ratio,
#     )

#     soft_cross = np.clip((align + 1.0) * 0.5, 0.0, 1.0) ** 2

#     prob = radial * (align_pos + cross_surface_gain * edge_factor * soft_cross)

#     if np.all(prob <= 1e-12):
#         prob = np.ones_like(prob, dtype=np.float64)

#     prob = prob.astype(np.float64)
#     prob /= prob.sum()
#     return prob


# # =========================================================
# # 13. choose valid centers
# # =========================================================
# def sample_valid_fixed_centers(
#     scene,
#     points,
#     normals,
#     candidate_ids,
#     reachable_mask,
#     largest_radius,
#     thickness,
#     min_center_dist,
#     num_centers=5,
#     max_trials=10000,
#     rng=None,
# ):
#     if rng is None:
#         rng = np.random.default_rng()

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

#         mask, info = extract_patch_with_visibility(
#             scene=scene,
#             points=points,
#             normals=normals,
#             reachable_mask=reachable_mask,
#             center_idx=idx,
#             radius=largest_radius,
#             thickness=thickness,
#         )

#         if mask is None:
#             continue

#         centers.append(idx)

#     if len(centers) < num_centers:
#         raise RuntimeError(
#             f"Only found {len(centers)} valid centers, need {num_centers}."
#         )

#     return np.array(centers, dtype=np.int64)


# # # =========================================================
# # # 14. try sample one finger robustly
# # # =========================================================
# # def sample_one_finger_points_strict(
# #     spc,
# #     scene,
# #     reachable_mask,
# #     candidate_center_ids,
# #     fixed_center_ids,
# #     finger_id,
# #     radius,
# #     thickness,
# #     points_per_finger=3000,
# #     max_center_retries=30,
# #     rng=None,
# # ):
# #     if rng is None:
# #         rng = np.random.default_rng()

# #     points = spc.points
# #     normals = spc.normals

# #     # 第一优先：先尝试当前固定 center
# #     center_try_list = [int(fixed_center_ids[finger_id])]

# #     # 再尝试重采 center
# #     if len(candidate_center_ids) > 0:
# #         extra = rng.choice(candidate_center_ids, size=min(max_center_retries, len(candidate_center_ids)), replace=len(candidate_center_ids) < max_center_retries)
# #         center_try_list.extend([int(x) for x in extra])

# #     best_candidate_ids = np.zeros((0,), dtype=np.int64)
# #     best_score = -1.0
# #     best_center_idx = int(fixed_center_ids[finger_id])

# #     # 两阶段阈值：先严格，再稍微放松
# #     setting_list = [
# #         dict(min_reachable_coverage=0.20, min_visible_coverage=0.30, min_final_points=80, normal_angle_deg=28.0),
# #         dict(min_reachable_coverage=0.12, min_visible_coverage=0.18, min_final_points=40, normal_angle_deg=35.0),
# #     ]

# #     for center_idx in center_try_list:
# #         for setting in setting_list:
# #             mask, info = extract_patch_with_visibility(
# #                 scene=scene,
# #                 points=points,
# #                 normals=normals,
# #                 reachable_mask=reachable_mask,
# #                 center_idx=center_idx,
# #                 radius=radius,
# #                 thickness=thickness,
# #                 normal_angle_deg=setting["normal_angle_deg"],
# #                 min_patch_points=150,
# #                 min_final_points=setting["min_final_points"],
# #                 min_reachable_coverage=setting["min_reachable_coverage"],
# #                 min_visible_coverage=setting["min_visible_coverage"],
# #             )

# #             candidate_ids = info["candidate_ids"]

# #             score = len(candidate_ids)
# #             if score > best_score:
# #                 best_score = score
# #                 best_candidate_ids = candidate_ids
# #                 best_center_idx = center_idx

# #             if mask is not None:
# #                 idx = np.where(mask)[0]
# #                 prob = compute_soft_contact_probability(
# #                     points=points,
# #                     normals=normals,
# #                     center_idx=center_idx,
# #                     candidate_ids=idx,
# #                     radius=radius,
# #                     cross_surface_gain=0.35,
# #                     edge_neighbor_ratio=0.35,
# #                 )

# #                 choose = rng.choice(
# #                     idx,
# #                     size=points_per_finger,
# #                     replace=(len(idx) < points_per_finger),
# #                     p=prob,
# #                 )

# #                 return (
# #                     points[choose].astype(np.float32),
# #                     np.full(points_per_finger, finger_id, dtype=np.int32),
# #                     np.full(points_per_finger, best_center_idx, dtype=np.int32),
# #                     best_center_idx,
# #                     True,
# #                 )

# #     # -------------------------
# #     # fallback 1:
# #     # 从 best_candidate_ids 局部重复采样补齐
# #     # -------------------------
# #     if len(best_candidate_ids) > 0:
# #         prob = compute_soft_contact_probability(
# #             points=points,
# #             normals=normals,
# #             center_idx=best_center_idx,
# #             candidate_ids=best_candidate_ids,
# #             radius=radius,
# #             cross_surface_gain=0.35,
# #             edge_neighbor_ratio=0.35,
# #         )

# #         choose = rng.choice(
# #             best_candidate_ids,
# #             size=points_per_finger,
# #             replace=True,
# #             p=prob,
# #         )

# #         return (
# #             points[choose].astype(np.float32),
# #             np.full(points_per_finger, finger_id, dtype=np.int32),
# #             np.full(points_per_finger, best_center_idx, dtype=np.int32),
# #             best_center_idx,
# #             False,
# #         )

# #     # -------------------------
# #     # fallback 2:
# #     # 当前 finger center 周围最近邻局部补点
# #     # -------------------------
# #     fallback_center = int(fixed_center_ids[finger_id])
# #     center_pt = points[fallback_center]
# #     dist = np.linalg.norm(points - center_pt[None, :], axis=1)
# #     local_idx = np.argsort(dist)[:max(points_per_finger, 200)]

# #     choose = rng.choice(local_idx, size=points_per_finger, replace=True)

# #     return (
# #         points[choose].astype(np.float32),
# #         np.full(points_per_finger, finger_id, dtype=np.int32),
# #         np.full(points_per_finger, fallback_center, dtype=np.int32),
# #         fallback_center,
# #         False,
# #     )


# # =========================================================
# # 15. one round, strictly fixed shape
# # =========================================================
# # def tactile_sampling_round_strict(
# #     spc,
# #     scene,
# #     reachable_mask,
# #     candidate_center_ids,
# #     fixed_center_ids,
# #     radius,
# #     thickness,
# #     points_per_finger=3000,
# #     rng=None,
# # ):
# #     if rng is None:
# #         rng = np.random.default_rng()

# #     all_pts = []
# #     all_fids = []
# #     all_center_ids = []
# #     new_center_ids = []

# #     num_fingers = len(fixed_center_ids)

# #     for fid in range(num_fingers):
# #         pts, fids, cids, final_center_idx, success = sample_one_finger_points_strict(
# #             spc=spc,
# #             scene=scene,
# #             reachable_mask=reachable_mask,
# #             candidate_center_ids=candidate_center_ids,
# #             fixed_center_ids=fixed_center_ids,
# #             finger_id=fid,
# #             radius=radius,
# #             thickness=thickness,
# #             points_per_finger=points_per_finger,
# #             max_center_retries=30,
# #             rng=rng,
# #         )

# #         all_pts.append(pts)
# #         all_fids.append(fids)
# #         all_center_ids.append(cids)
# #         new_center_ids.append(final_center_idx)

# #     pts = np.vstack(all_pts).astype(np.float32)
# #     fids = np.concatenate(all_fids).astype(np.int32)
# #     cids = np.concatenate(all_center_ids).astype(np.int32)
# #     new_center_ids = np.asarray(new_center_ids, dtype=np.int32)

# #     return pts, fids, cids, new_center_ids
# import numpy as np


# # =========================================================
# # A. non-overlap helpers
# # =========================================================
# def compute_patch_overlap_ratio(mask_a, mask_b):
#     inter = int(np.count_nonzero(mask_a & mask_b))
#     if inter == 0:
#         return 0.0

#     denom = max(
#         min(int(np.count_nonzero(mask_a)), int(np.count_nonzero(mask_b))),
#         1
#     )
#     return float(inter / denom)


# def extract_patch_candidates_fixed_center(
#     scene,
#     points,
#     normals,
#     reachable_mask,
#     center_idx,
#     radius,
#     thickness,
# ):
#     """
#     Try extracting a valid visible patch for one fixed center.
#     If strict settings fail, progressively relax constraints.
#     If all fail, fall back to KNN around center.
#     """
#     setting_list = [
#         dict(
#             normal_angle_deg=28.0,
#             min_patch_points=150,
#             min_final_points=80,
#             min_reachable_coverage=0.20,
#             min_visible_coverage=0.30,
#         ),
#         dict(
#             normal_angle_deg=35.0,
#             min_patch_points=80,
#             min_final_points=30,
#             min_reachable_coverage=0.10,
#             min_visible_coverage=0.15,
#         ),
#     ]

#     best_ids = np.zeros((0,), dtype=np.int64)
#     best_info = {
#         "reason": "no_candidate",
#         "coverage_reachable": 0.0,
#         "coverage_visible": 0.0,
#         "n_patch": 0,
#         "n_reachable": 0,
#         "n_visible": 0,
#         "source": "none",
#         "candidate_ids": [],
#     }

#     for setting in setting_list:
#         mask, info = extract_patch_with_visibility(
#             scene=scene,
#             points=points,
#             normals=normals,
#             reachable_mask=reachable_mask,
#             center_idx=center_idx,
#             radius=radius,
#             thickness=thickness,
#             normal_angle_deg=setting["normal_angle_deg"],
#             min_patch_points=setting["min_patch_points"],
#             min_final_points=setting["min_final_points"],
#             min_reachable_coverage=setting["min_reachable_coverage"],
#             min_visible_coverage=setting["min_visible_coverage"],
#         )

#         if mask is not None:
#             ids = np.where(mask)[0].astype(np.int64)
#             info = dict(info)
#             info["source"] = "visible_patch"
#             return ids, info

#         info = dict(info)
#         candidate_ids = np.asarray(info.get("candidate_ids", []), dtype=np.int64)
#         if len(candidate_ids) > len(best_ids):
#             best_ids = candidate_ids
#             best_info = dict(info)
#             best_info["source"] = "relaxed_candidates"

#     if len(best_ids) > 0:
#         return best_ids, best_info

#     center_pt = points[center_idx]
#     dist = np.linalg.norm(points - center_pt[None, :], axis=1)
#     knn_ids = np.argsort(dist)[:128].astype(np.int64)

#     fallback_info = dict(best_info)
#     fallback_info["source"] = "center_knn_fallback"
#     fallback_info["n_visible"] = len(knn_ids)
#     fallback_info["candidate_ids"] = knn_ids.tolist()
#     return knn_ids, fallback_info


# def resolve_multi_finger_candidates(points, center_ids, candidate_ids_per_finger):
#     """
#     Resolve candidate overlap globally:
#     each point is assigned to the nearest finger center among all fingers
#     that included it as a candidate.
#     """
#     owner = np.full(len(points), -1, dtype=np.int32)
#     best_dist = np.full(len(points), np.inf, dtype=np.float32)

#     for fid, center_idx in enumerate(center_ids):
#         ids = np.asarray(candidate_ids_per_finger[fid], dtype=np.int64)
#         if len(ids) == 0:
#             continue

#         dist = np.linalg.norm(points[ids] - points[center_idx][None, :], axis=1)
#         better = dist < best_dist[ids]

#         owner[ids[better]] = fid
#         best_dist[ids[better]] = dist[better]

#     exclusive_ids_per_finger = []
#     for fid in range(len(center_ids)):
#         ids = np.where(owner == fid)[0].astype(np.int64)
#         exclusive_ids_per_finger.append(ids)

#     return exclusive_ids_per_finger


# # =========================================================
# # 13. choose valid fixed centers (non-overlapping)
# # =========================================================
# def sample_valid_fixed_centers(
#     scene,
#     points,
#     normals,
#     candidate_ids,
#     reachable_mask,
#     largest_radius,
#     thickness,
#     min_center_dist,
#     num_centers=5,
#     max_trials=10000,
#     max_patch_overlap=0.0,
#     rng=None,
# ):
#     """
#     Sample fixed finger centers on the object surface.
#     Requirements:
#     1) centers must be separated by min_center_dist
#     2) their largest-radius patches must not overlap beyond max_patch_overlap
#     """
#     if rng is None:
#         rng = np.random.default_rng()

#     candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
#     if len(candidate_ids) == 0:
#         raise RuntimeError("candidate_ids is empty in sample_valid_fixed_centers.")

#     centers = []
#     accepted_masks = []
#     trials = 0

#     while len(centers) < num_centers and trials < max_trials:
#         trials += 1
#         idx = int(rng.choice(candidate_ids))

#         # enforce center distance
#         if any(np.linalg.norm(points[idx] - points[c]) < min_center_dist for c in centers):
#             continue

#         mask, info = extract_patch_with_visibility(
#             scene=scene,
#             points=points,
#             normals=normals,
#             reachable_mask=reachable_mask,
#             center_idx=idx,
#             radius=largest_radius,
#             thickness=thickness,
#         )

#         if mask is None:
#             continue

#         overlap_too_large = any(
#             compute_patch_overlap_ratio(mask, prev_mask) > max_patch_overlap
#             for prev_mask in accepted_masks
#         )
#         if overlap_too_large:
#             continue

#         centers.append(idx)
#         accepted_masks.append(mask)

#         print(
#             f"[CENTER] accept center {len(centers)-1}: idx={idx}, "
#             f"patch={info.get('n_patch', 0)}, "
#             f"reachable={info.get('n_reachable', 0)}, "
#             f"visible={info.get('n_visible', 0)}"
#         )

#     if len(centers) < num_centers:
#         raise RuntimeError(
#             f"Only found {len(centers)} non-overlapping centers, need {num_centers}. "
#             f"Try increasing surface_sample_n / num_rays, or reducing start_ratio."
#         )

#     return np.asarray(centers, dtype=np.int64)


# # =========================================================
# # 14. one round with non-overlapping finger areas
# # =========================================================
# def tactile_sampling_round_nonoverlap(
#     spc,
#     scene,
#     reachable_mask,
#     center_ids,
#     radius,
#     thickness,
#     points_per_finger=3000,
#     rng=None,
# ):
#     """
#     For one radius round:
#     1) extract candidate patch per finger
#     2) resolve overlapping candidates globally
#     3) sample a fixed number of points per finger
#     """
#     if rng is None:
#         rng = np.random.default_rng()

#     points = spc.points
#     normals = spc.normals

#     candidate_ids_per_finger = []
#     info_per_finger = []

#     # step 1: collect candidates for each finger
#     for fid, center_idx in enumerate(center_ids):
#         candidate_ids, info = extract_patch_candidates_fixed_center(
#             scene=scene,
#             points=points,
#             normals=normals,
#             reachable_mask=reachable_mask,
#             center_idx=center_idx,
#             radius=radius,
#             thickness=thickness,
#         )
#         candidate_ids_per_finger.append(candidate_ids)
#         info_per_finger.append(info)

#     # step 2: resolve overlap globally
#     exclusive_ids_per_finger = resolve_multi_finger_candidates(
#         points=points,
#         center_ids=center_ids,
#         candidate_ids_per_finger=candidate_ids_per_finger,
#     )

#     # step 3: per finger sampling
#     all_pts = []
#     all_fids = []
#     all_center_ids = []

#     for fid, center_idx in enumerate(center_ids):
#         candidate_ids = exclusive_ids_per_finger[fid]

#         if len(candidate_ids) == 0:
#             candidate_ids = np.asarray([center_idx], dtype=np.int64)

#         prob = compute_soft_contact_probability(
#             points=points,
#             normals=normals,
#             center_idx=center_idx,
#             candidate_ids=candidate_ids,
#             radius=radius,
#             cross_surface_gain=0.35,
#             edge_neighbor_ratio=0.35,
#         )

#         choose = rng.choice(
#             candidate_ids,
#             size=points_per_finger,
#             replace=len(candidate_ids) < points_per_finger,
#             p=prob,
#         )

#         all_pts.append(points[choose])
#         all_fids.append(np.full(points_per_finger, fid, dtype=np.int32))
#         all_center_ids.append(np.full(points_per_finger, center_idx, dtype=np.int32))

#         print(
#             f"[Finger {fid}] source={info_per_finger[fid]['source']} | "
#             f"raw={len(candidate_ids_per_finger[fid])} | "
#             f"exclusive={len(candidate_ids)} | sampled={points_per_finger}"
#         )

#     pts = np.vstack(all_pts).astype(np.float32)
#     fids = np.concatenate(all_fids).astype(np.int32)
#     cids = np.concatenate(all_center_ids).astype(np.int32)

#     return pts, fids, cids


# # =========================================================
# # 16. generate one tactile observation
# # =========================================================
# def generate_tactile_touch_points(
#     mesh,
#     surface_sample_n=200000,
#     num_rays=20000,
#     beam_radius=0.1,
#     rounds=5,
#     start_ratio=0.12,
#     end_ratio=0.03,
#     thickness_ratio=0.01,
#     points_per_finger=3000,
#     num_fingers=5,
#     rng=None,
# ):
#     """
#     Generate tactile touch points for multiple fingers over multiple rounds.

#     Guarantees:
#     - finger surface regions are mutually exclusive on the 3D surface
#     - fixed centers are reused across all rounds
#     """
#     if rng is None:
#         rng = np.random.default_rng()

#     # sample surface points + normals
#     spc = sample_from_mesh(mesh, n=surface_sample_n)
#     diag = bbox_diag(spc.points)

#     # get reachable candidate region from outside ray hits
#     candidate_ids, hit_points = raycast_outer_hits(
#         mesh,
#         spc.points,
#         num_rays=num_rays,
#         rng=rng,
#     )

#     reachable_mask = build_global_reachable_mask(
#         spc.points,
#         hit_points,
#         beam_radius=beam_radius,
#     )

#     scene = build_raycast_scene(mesh)

#     largest_radius = start_ratio * diag
#     thickness = thickness_ratio * diag

#     # choose fixed non-overlapping centers once
#     fixed_center_ids = sample_valid_fixed_centers(
#         scene=scene,
#         points=spc.points,
#         normals=spc.normals,
#         candidate_ids=candidate_ids,
#         reachable_mask=reachable_mask,
#         largest_radius=largest_radius,
#         thickness=thickness,
#         min_center_dist=1.5 * largest_radius,
#         num_centers=num_fingers,
#         max_trials=10000,
#         max_patch_overlap=0.0,
#         rng=rng,
#     )

#     print("[INFO] fixed non-overlapping center ids:", fixed_center_ids.tolist())

#     radius_schedule = np.linspace(start_ratio, end_ratio, rounds)

#     round_points = []
#     round_ids = []
#     finger_ids = []
#     center_ids_per_point = []

#     for rid, ratio in enumerate(radius_schedule):
#         radius = float(ratio) * diag

#         print(f"\n[ROUND {rid}] radius_ratio={ratio:.4f}, radius={radius:.6f}")

#         pts, fids, cids = tactile_sampling_round_nonoverlap(
#             spc=spc,
#             scene=scene,
#             reachable_mask=reachable_mask,
#             center_ids=fixed_center_ids,
#             radius=radius,
#             thickness=thickness,
#             points_per_finger=points_per_finger,
#             rng=rng,
#         )

#         round_points.append(pts)
#         round_ids.append(np.full(len(pts), rid, dtype=np.int32))
#         finger_ids.append(fids)
#         center_ids_per_point.append(cids)

#     touch_points = np.vstack(round_points).astype(np.float32)
#     touch_round_ids = np.concatenate(round_ids).astype(np.int32)
#     touch_finger_ids = np.concatenate(finger_ids).astype(np.int32)
#     touch_center_ids = np.concatenate(center_ids_per_point).astype(np.int32)
#     touch_centers = spc.points[fixed_center_ids].astype(np.float32)

#     expected_n = rounds * num_fingers * points_per_finger
#     if len(touch_points) != expected_n:
#         raise RuntimeError(
#             f"Tactile point count mismatch: got {len(touch_points)}, expected {expected_n}"
#         )

#     return {
#         "touch_points": touch_points,
#         "touch_round_ids": touch_round_ids,
#         "touch_finger_ids": touch_finger_ids,
#         "touch_center_ids": touch_center_ids,
#         "touch_centers": touch_centers,
#     }

# # =========================================================
# # 16. generate one tactile observation
# # strictly fixed Nt = rounds * fingers * points_per_finger
# # =========================================================
# def generate_tactile_touch_points(
#     mesh,
#     surface_sample_n=200000,
#     num_rays=20000,
#     beam_radius=0.1,
#     rounds=5,
#     start_ratio=0.12,
#     end_ratio=0.03,
#     thickness_ratio=0.01,
#     points_per_finger=3000,
#     num_fingers=5,
#     rng=None,
# ):
#     if rng is None:
#         rng = np.random.default_rng()

#     spc = sample_from_mesh(mesh, n=surface_sample_n)
#     diag = bbox_diag(spc.points)

#     candidate_ids, hit_points = raycast_outer_hits(mesh, spc.points, num_rays=num_rays, rng=rng)
#     reachable_mask = build_global_reachable_mask(spc.points, hit_points, beam_radius=beam_radius)
#     scene = build_raycast_scene(mesh)

#     largest_radius = start_ratio * diag
#     smallest_radius = end_ratio * diag
#     thickness = thickness_ratio * diag

#     fixed_center_ids = sample_valid_fixed_centers(
#         scene=scene,
#         points=spc.points,
#         normals=spc.normals,
#         candidate_ids=candidate_ids,
#         reachable_mask=reachable_mask,
#         largest_radius=largest_radius,
#         thickness=thickness,
#         min_center_dist=2 * smallest_radius,
#         num_centers=num_fingers,
#         max_trials=10000,
#         rng=rng,
#     )

#     radius_schedule = np.linspace(start_ratio, end_ratio, rounds)

#     round_points = []
#     round_ids = []
#     finger_ids = []
#     center_ids_per_point = []

#     current_center_ids = fixed_center_ids.copy()

#     for rid, ratio in enumerate(radius_schedule):
#         radius = float(ratio) * diag

#         pts, fids, cids, updated_center_ids = tactile_sampling_round_nonoverlap(
#             spc=spc,
#             scene=scene,
#             reachable_mask=reachable_mask,
#             # candidate_center_ids=candidate_ids,
#              center_ids=fixed_center_ids,   # ✅ 改这里
#             fixed_center_ids=current_center_ids,
#             radius=radius,
#             thickness=thickness,
#             points_per_finger=points_per_finger,
#             rng=rng,
#         )

#         round_points.append(pts)
#         round_ids.append(np.full(len(pts), rid, dtype=np.int32))
#         finger_ids.append(fids)
#         center_ids_per_point.append(cids)

#         # 更新每个 finger 当前可用 center
#         current_center_ids = updated_center_ids.copy()

#     touch_points = np.vstack(round_points).astype(np.float32)
#     touch_round_ids = np.concatenate(round_ids).astype(np.int32)
#     touch_finger_ids = np.concatenate(finger_ids).astype(np.int32)
#     touch_center_ids = np.concatenate(center_ids_per_point).astype(np.int32)
#     touch_centers = spc.points[current_center_ids].astype(np.float32)

#     expected_n = rounds * num_fingers * points_per_finger
#     if len(touch_points) != expected_n:
#         raise RuntimeError(
#             f"Tactile point count mismatch: got {len(touch_points)}, expected {expected_n}"
#         )

#     return {
#         "touch_points": touch_points,
#         "touch_round_ids": touch_round_ids,
#         "touch_finger_ids": touch_finger_ids,
#         "touch_center_ids": touch_center_ids,
#         "touch_centers": touch_centers,
#     }


# # =========================================================
# # 17. one mesh -> one merged npz
# # =========================================================
# def process_single_obj_to_merged_npz(
#     obj_path,
#     out_path,
#     num_tactile_samples=10,
#     tactile_surface_sample_n=200000,
#     tactile_num_rays=20000,
#     tactile_beam_radius=0.1,
#     tactile_rounds=5,
#     tactile_start_ratio=0.12,
#     tactile_end_ratio=0.03,
#     tactile_thickness_ratio=0.01,
#     tactile_points_per_finger=3000,
#     tactile_num_fingers=5,
#     num_surface_points=235000,
#     num_query_points=250000,
# ):
#     print("\n==================================================")
#     print("[PROCESS]", obj_path)
#     print("==================================================")

#     mesh_name = os.path.splitext(os.path.basename(obj_path))[0]

#     mesh = trimesh.load(obj_path, force="mesh")
#     mesh.process(validate=True)
#     mesh = scale_to_unit_sphere(mesh)

#     # -------------------------
#     # geometry supervision
#     # -------------------------
#     surface_points, surface_normals = sample_surface_points_for_storage(
#         mesh,
#         num_surface_points=num_surface_points
#     )

#     print("[INFO] sampling query points using sample_sdf_near_surface ...")
#     query_points, query_sdf = sample_sdf_near_surface(
#         mesh,
#         number_of_points=num_query_points
#     )
#     query_points = query_points.astype(np.float32)
#     query_sdf = query_sdf.astype(np.float32)

#     # -------------------------
#     # multiple tactile observations
#     # -------------------------
#     touch_points_all = []
#     touch_round_ids_all = []
#     touch_finger_ids_all = []
#     touch_center_ids_all = []
#     touch_centers_all = []

#     for sample_idx in range(num_tactile_samples):
#         print(f"[INFO] tactile sample {sample_idx + 1}/{num_tactile_samples}")

#         rng = np.random.default_rng()

#         tactile_data = generate_tactile_touch_points(
#             mesh=mesh,
#             surface_sample_n=tactile_surface_sample_n,
#             num_rays=tactile_num_rays,
#             beam_radius=tactile_beam_radius,
#             rounds=tactile_rounds,
#             start_ratio=tactile_start_ratio,
#             end_ratio=tactile_end_ratio,
#             thickness_ratio=tactile_thickness_ratio,
#             points_per_finger=tactile_points_per_finger,
#             num_fingers=tactile_num_fingers,
#             rng=rng,
#         )

#         touch_points_all.append(tactile_data["touch_points"])
#         touch_round_ids_all.append(tactile_data["touch_round_ids"])
#         touch_finger_ids_all.append(tactile_data["touch_finger_ids"])
#         touch_center_ids_all.append(tactile_data["touch_center_ids"])
#         touch_centers_all.append(tactile_data["touch_centers"])

#     # stack is now safe because shapes are strictly fixed
#     touch_points_all = np.stack(touch_points_all, axis=0).astype(np.float32)
#     touch_round_ids_all = np.stack(touch_round_ids_all, axis=0).astype(np.int32)
#     touch_finger_ids_all = np.stack(touch_finger_ids_all, axis=0).astype(np.int32)
#     touch_center_ids_all = np.stack(touch_center_ids_all, axis=0).astype(np.int32)
#     touch_centers_all = np.stack(touch_centers_all, axis=0).astype(np.float32)

#     os.makedirs(os.path.dirname(out_path), exist_ok=True)

#     np.savez_compressed(
#         out_path,
#         surface_points=surface_points,
#         surface_normals=surface_normals,
#         query_points=query_points,
#         query_sdf=query_sdf,
#         touch_points=touch_points_all,
#         touch_round_ids=touch_round_ids_all,
#         touch_finger_ids=touch_finger_ids_all,
#         touch_center_ids=touch_center_ids_all,
#         touch_centers=touch_centers_all,
#         mesh_name=np.array(mesh_name),
#         num_tactile_samples=np.array(num_tactile_samples, dtype=np.int32),
#     )

#     print("[SAVED]", out_path)
#     print("surface_points   :", surface_points.shape)
#     print("surface_normals  :", surface_normals.shape)
#     print("query_points     :", query_points.shape)
#     print("query_sdf        :", query_sdf.shape)
#     print("touch_points     :", touch_points_all.shape)
#     print("touch_round_ids  :", touch_round_ids_all.shape)
#     print("touch_finger_ids :", touch_finger_ids_all.shape)
#     print("touch_center_ids :", touch_center_ids_all.shape)
#     print("touch_centers    :", touch_centers_all.shape)


# # =========================================================
# # 18. process split
# # =========================================================
# def process_split(
#     category_dir,
#     split="train",
#     max_objects=80,
#     num_tactile_samples=10,
#     output_folder_name=None,
# ):
#     obj_dir = os.path.join(category_dir, f"{split}_obj")

#     if output_folder_name is None:
#         output_folder_name = f"tactistruct_npz_{split}"

#     out_dir = os.path.join(category_dir, output_folder_name)

#     if not os.path.isdir(obj_dir):
#         print(f"[WARN] skip missing dir: {obj_dir}")
#         return

#     os.makedirs(out_dir, exist_ok=True)

#     obj_files = sorted([
#         f for f in os.listdir(obj_dir)
#         if f.lower().endswith(".obj")
#     ])[:max_objects]

#     print(f"[INFO] {split}: found {len(obj_files)} obj files in {obj_dir}")

#     for name in obj_files:
#         obj_path = os.path.join(obj_dir, name)
#         out_path = os.path.join(out_dir, os.path.splitext(name)[0] + ".npz")

#         if os.path.exists(out_path):
#             print("[SKIP exists]", out_path)
#             continue

#         try:
#             process_single_obj_to_merged_npz(
#                 obj_path=obj_path,
#                 out_path=out_path,
#                 num_tactile_samples=num_tactile_samples,
#             )
#         except Exception as e:
#             print("[FAILED]", obj_path)
#             print("Error:", e)


# # =========================================================
# # 19. process all categories
# # =========================================================
# def process_all_categories(
#     root_dir,
#     split="train",
#     max_objects_per_category=80,
#     num_tactile_samples=10,
#     category_names=None,
# ):
#     subdirs = sorted([
#         os.path.join(root_dir, d)
#         for d in os.listdir(root_dir)
#         if os.path.isdir(os.path.join(root_dir, d))
#     ])

#     if category_names is not None:
#         category_names = set(category_names)
#         subdirs = [d for d in subdirs if os.path.basename(d) in category_names]

#     if not subdirs:
#         print("[WARN] no category folders found under:", root_dir)
#         return

#     print(f"[INFO] found {len(subdirs)} category folders under root.")

#     for category_dir in subdirs:
#         category_name = os.path.basename(category_dir)
#         print(f"\n########## Processing category: {category_name} ##########")

#         process_split(
#             category_dir=category_dir,
#             split=split,
#             max_objects=max_objects_per_category,
#             num_tactile_samples=num_tactile_samples,
#             output_folder_name=f"tactistruct_npz_{split}",
#         )


# # =========================================================
# # 20. main
# # =========================================================
# if __name__ == "__main__":
#     root_dir = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40"

#     process_all_categories(
#         root_dir=root_dir,
#         split="train",
#         max_objects_per_category=80,
#         num_tactile_samples=10,
#         category_names=["sofa"]
#     )

#     print("\nAll done.")










import os
import numpy as np
import trimesh
import open3d as o3d

from scipy.spatial import cKDTree
from mesh_to_sdf import sample_sdf_near_surface


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
# 3. dense surface sampling
# =========================================================
def sample_from_mesh(mesh, n=200000):
    points, face_idx = mesh.sample(n, return_index=True)

    normals = mesh.face_normals[face_idx].astype(np.float32)
    normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12, None)

    return SurfacePointCloud(
        mesh=mesh,
        points=points.astype(np.float32),
        normals=normals.astype(np.float32),
    )


# =========================================================
# 4. explicit surface storage
# =========================================================
def sample_surface_points_for_storage(mesh, num_surface_points=235000):
    surface_points, face_idx = mesh.sample(num_surface_points, return_index=True)
    surface_points = surface_points.astype(np.float32)

    surface_normals = mesh.face_normals[face_idx].astype(np.float32)
    surface_normals /= np.clip(
        np.linalg.norm(surface_normals, axis=1, keepdims=True),
        1e-8,
        None
    )

    return surface_points, surface_normals


# =========================================================
# 5. bbox diagonal
# =========================================================
def bbox_diag(points):
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    return float(np.linalg.norm(mx - mn))


# =========================================================
# 6. raycasting scene
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
# 7. ray casting outer hits
# =========================================================
def raycast_outer_hits(mesh, points, num_rays=20000, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    print("[INFO] ray casting")

    scene = build_raycast_scene(mesh)

    dirs = rng.normal(size=(num_rays, 3))
    dirs /= np.clip(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12, None)

    origins = dirs * 3.0
    directions = -dirs

    rays = np.concatenate([origins, directions], axis=1).astype(np.float32)
    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

    ans = scene.cast_rays(rays)
    t_hit = ans["t_hit"].numpy()

    valid = np.isfinite(t_hit)
    hit_points = origins[valid] + directions[valid] * t_hit[valid][:, None]

    print("[INFO] raw ray hits:", len(hit_points))

    if len(hit_points) == 0:
        raise RuntimeError("No outer ray hits found.")

    tree = cKDTree(points)
    _, ids = tree.query(hit_points, k=1)
    candidate_ids = np.unique(ids.astype(np.int64))

    print("[INFO] unique outer surface candidates:", len(candidate_ids))
    return candidate_ids, hit_points.astype(np.float32)


# =========================================================
# 8. reachable region
# =========================================================
def build_global_reachable_mask(points, hit_points, beam_radius):
    print("[INFO] building reachable region")
    hit_tree = cKDTree(hit_points)
    dist, _ = hit_tree.query(points, k=1, distance_upper_bound=beam_radius)
    reachable_mask = np.isfinite(dist)
    print("[INFO] reachable points:", int(reachable_mask.sum()))
    return reachable_mask


# =========================================================
# 9. theoretical tangent patch
# =========================================================
def extract_theoretical_patch(
    points,
    normals,
    center_idx,
    radius,
    thickness,
    normal_angle_deg=28.0,
    min_patch_points=150,
):
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

    if int(mask.sum()) < int(min_patch_points):
        return None

    return mask


# =========================================================
# 10. local visibility
# =========================================================
def filter_visible_points(
    scene,
    center,
    center_normal,
    candidate_points,
    eps=2e-3,
    hit_tol=2e-3,
):
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
# 11. patch + reachable + visibility
# =========================================================
def extract_patch_with_visibility(
    scene,
    points,
    normals,
    reachable_mask,
    center_idx,
    radius,
    thickness,
    normal_angle_deg=28.0,
    min_patch_points=150,
    min_final_points=80,
    min_reachable_coverage=0.2,
    min_visible_coverage=0.3,
    eps=1e-3,
    hit_tol=3e-3,
):
    patch_mask = extract_theoretical_patch(
        points=points,
        normals=normals,
        center_idx=center_idx,
        radius=radius,
        thickness=thickness,
        normal_angle_deg=normal_angle_deg,
        min_patch_points=min_patch_points,
    )

    if patch_mask is None:
        return None, {
            "reason": "theoretical_patch_too_small",
            "coverage_reachable": 0.0,
            "coverage_visible": 0.0,
            "n_patch": 0,
            "n_reachable": 0,
            "n_visible": 0,
            "candidate_ids": np.zeros((0,), dtype=np.int64),
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
            "n_visible": 0,
            "candidate_ids": reachable_ids.astype(np.int64),
        }

    if len(reachable_ids) < min_final_points:
        return None, {
            "reason": "reachable_points_too_few",
            "coverage_reachable": coverage_reachable,
            "coverage_visible": 0.0,
            "n_patch": len(patch_ids),
            "n_reachable": len(reachable_ids),
            "n_visible": 0,
            "candidate_ids": reachable_ids.astype(np.int64),
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
        hit_tol=hit_tol,
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
            "n_visible": len(visible_ids),
            "candidate_ids": visible_ids.astype(np.int64),
        }

    if len(visible_ids) < min_final_points:
        return None, {
            "reason": "visible_points_too_few",
            "coverage_reachable": coverage_reachable,
            "coverage_visible": coverage_visible,
            "n_patch": len(patch_ids),
            "n_reachable": len(reachable_ids),
            "n_visible": len(visible_ids),
            "candidate_ids": visible_ids.astype(np.int64),
        }

    final_mask = np.zeros(len(points), dtype=bool)
    final_mask[visible_ids] = True

    return final_mask, {
        "reason": "ok",
        "coverage_reachable": coverage_reachable,
        "coverage_visible": coverage_visible,
        "n_patch": len(patch_ids),
        "n_reachable": len(reachable_ids),
        "n_visible": len(visible_ids),
        "candidate_ids": visible_ids.astype(np.int64),
    }


# =========================================================
# 12. edge-aware probability
# =========================================================
def estimate_edge_deformation_factor(
    points,
    normals,
    center_idx,
    radius,
    edge_neighbor_ratio=0.35,
):
    c = points[center_idx]
    n0 = normals[center_idx]

    d = np.linalg.norm(points - c[None, :], axis=1)
    local_mask = d <= (edge_neighbor_ratio * radius)
    local_ids = np.where(local_mask)[0]

    if len(local_ids) < 20:
        return 0.0

    align = normals[local_ids] @ n0
    align = np.clip(align, -1.0, 1.0)

    normal_variation = 1.0 - np.mean(np.abs(align))
    edge_factor = np.clip(normal_variation / 0.5, 0.0, 1.0)
    return float(edge_factor)


def compute_soft_contact_probability(
    points,
    normals,
    center_idx,
    candidate_ids,
    radius,
    cross_surface_gain=0.35,
    edge_neighbor_ratio=0.35,
):
    if len(candidate_ids) == 0:
        return np.zeros((0,), dtype=np.float64)

    c = points[center_idx]
    n0 = normals[center_idx]

    pts = points[candidate_ids]
    nrm = normals[candidate_ids]

    v = pts - c[None, :]
    dist = np.linalg.norm(v, axis=1)

    radial = 1.0 - dist / max(radius, 1e-8)
    radial = np.clip(radial, 0.0, None) ** 2

    align = nrm @ n0
    align_pos = np.clip(align, 0.0, 1.0)

    edge_factor = estimate_edge_deformation_factor(
        points=points,
        normals=normals,
        center_idx=center_idx,
        radius=radius,
        edge_neighbor_ratio=edge_neighbor_ratio,
    )

    soft_cross = np.clip((align + 1.0) * 0.5, 0.0, 1.0) ** 2

    prob = radial * (align_pos + cross_surface_gain * edge_factor * soft_cross)

    if np.all(prob <= 1e-12):
        prob = np.ones_like(prob, dtype=np.float64)

    prob = prob.astype(np.float64)
    prob /= prob.sum()
    return prob


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
        "candidate_ids": [],
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

        info = dict(info)
        candidate_ids = np.asarray(info.get("candidate_ids", []), dtype=np.int64)
        if len(candidate_ids) > len(best_ids):
            best_ids = candidate_ids
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
    fallback_info["candidate_ids"] = knn_ids.tolist()
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
# 13. choose valid fixed centers (non-overlapping)
# =========================================================
def sample_valid_fixed_centers(
    scene,
    points,
    normals,
    candidate_ids,
    reachable_mask,
    largest_radius,
    thickness,
    min_center_dist,
    num_centers=5,
    max_trials=10000,
    max_patch_overlap=0.0,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    if len(candidate_ids) == 0:
        raise RuntimeError("candidate_ids is empty in sample_valid_fixed_centers.")

    centers = []
    accepted_masks = []
    trials = 0

    while len(centers) < num_centers and trials < max_trials:
        trials += 1
        idx = int(rng.choice(candidate_ids))

        if any(np.linalg.norm(points[idx] - points[c]) < min_center_dist for c in centers):
            continue

        mask, info = extract_patch_with_visibility(
            scene=scene,
            points=points,
            normals=normals,
            reachable_mask=reachable_mask,
            center_idx=idx,
            radius=largest_radius,
            thickness=thickness,
        )

        if mask is None:
            continue

        overlap_too_large = any(
            compute_patch_overlap_ratio(mask, prev_mask) > max_patch_overlap
            for prev_mask in accepted_masks
        )
        if overlap_too_large:
            continue

        centers.append(idx)
        accepted_masks.append(mask)

        print(
            f"[CENTER] accept center {len(centers)-1}: idx={idx}, "
            f"patch={info.get('n_patch', 0)}, "
            f"reachable={info.get('n_reachable', 0)}, "
            f"visible={info.get('n_visible', 0)}"
        )

    if len(centers) < num_centers:
        raise RuntimeError(
            f"Only found {len(centers)} non-overlapping centers, need {num_centers}. "
            f"Try increasing surface_sample_n / num_rays, or reducing start_ratio."
        )

    return np.asarray(centers, dtype=np.int64)


# =========================================================
# 14. one round with non-overlapping finger areas
# =========================================================
def tactile_sampling_round_nonoverlap(
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


# =========================================================
# 15. generate one tactile observation
# =========================================================
def generate_tactile_touch_points(
    mesh,
    surface_sample_n=200000,
    num_rays=20000,
    beam_radius=0.1,
    rounds=5,
    start_ratio=0.12,
    end_ratio=0.03,
    thickness_ratio=0.01,
    points_per_finger=3000,
    num_fingers=5,
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

    scene = build_raycast_scene(mesh)

    largest_radius = start_ratio * diag
    thickness = thickness_ratio * diag

    fixed_center_ids = sample_valid_fixed_centers(
        scene=scene,
        points=spc.points,
        normals=spc.normals,
        candidate_ids=candidate_ids,
        reachable_mask=reachable_mask,
        largest_radius=largest_radius,
        thickness=thickness,
        min_center_dist=1.5 * largest_radius,
        num_centers=num_fingers,
        max_trials=10000,
        max_patch_overlap=0.0,
        rng=rng,
    )

    print("[INFO] fixed non-overlapping center ids:", fixed_center_ids.tolist())

    radius_schedule = np.linspace(start_ratio, end_ratio, rounds)

    round_points = []
    round_ids = []
    finger_ids = []
    center_ids_per_point = []

    for rid, ratio in enumerate(radius_schedule):
        radius = float(ratio) * diag

        print(f"\n[ROUND {rid}] radius_ratio={ratio:.4f}, radius={radius:.6f}")

        pts, fids, cids = tactile_sampling_round_nonoverlap(
            spc=spc,
            scene=scene,
            reachable_mask=reachable_mask,
            center_ids=fixed_center_ids,
            radius=radius,
            thickness=thickness,
            points_per_finger=points_per_finger,
            rng=rng,
        )

        round_points.append(pts)
        round_ids.append(np.full(len(pts), rid, dtype=np.int32))
        finger_ids.append(fids)
        center_ids_per_point.append(cids)

    touch_points = np.vstack(round_points).astype(np.float32)
    touch_round_ids = np.concatenate(round_ids).astype(np.int32)
    touch_finger_ids = np.concatenate(finger_ids).astype(np.int32)
    touch_center_ids = np.concatenate(center_ids_per_point).astype(np.int32)
    touch_centers = spc.points[fixed_center_ids].astype(np.float32)

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
    }


# =========================================================
# 16. one mesh -> one merged npz
# =========================================================
def process_single_obj_to_merged_npz(
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
    tactile_num_fingers=5,
    num_surface_points=235000,
    num_query_points=250000,
):
    print("\n==================================================")
    print("[PROCESS]", obj_path)
    print("==================================================")

    mesh_name = os.path.splitext(os.path.basename(obj_path))[0]

    mesh = trimesh.load(obj_path, force="mesh")
    mesh.process(validate=True)
    mesh = scale_to_unit_sphere(mesh)

    surface_points, surface_normals = sample_surface_points_for_storage(
        mesh,
        num_surface_points=num_surface_points
    )

    print("[INFO] sampling query points using sample_sdf_near_surface ...")
    query_points, query_sdf = sample_sdf_near_surface(
        mesh,
        number_of_points=num_query_points
    )
    query_points = query_points.astype(np.float32)
    query_sdf = query_sdf.astype(np.float32)

    touch_points_all = []
    touch_round_ids_all = []
    touch_finger_ids_all = []
    touch_center_ids_all = []
    touch_centers_all = []

    for sample_idx in range(num_tactile_samples):
        print(f"[INFO] tactile sample {sample_idx + 1}/{num_tactile_samples}")

        rng = np.random.default_rng()

        tactile_data = generate_tactile_touch_points(
            mesh=mesh,
            surface_sample_n=tactile_surface_sample_n,
            num_rays=tactile_num_rays,
            beam_radius=tactile_beam_radius,
            rounds=tactile_rounds,
            start_ratio=tactile_start_ratio,
            end_ratio=tactile_end_ratio,
            thickness_ratio=tactile_thickness_ratio,
            points_per_finger=tactile_points_per_finger,
            num_fingers=tactile_num_fingers,
            rng=rng,
        )

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
        query_points=query_points,
        query_sdf=query_sdf,
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
    print("touch_finger_ids :", touch_finger_ids_all.shape)
    print("touch_center_ids :", touch_center_ids_all.shape)
    print("touch_centers    :", touch_centers_all.shape)


# =========================================================
# 17. process split
# =========================================================
def process_split(
    category_dir,
    split="train",
    max_objects=80,
    num_tactile_samples=10,
    output_folder_name=None,
):
    obj_dir = os.path.join(category_dir, f"{split}_obj")

    if output_folder_name is None:
        output_folder_name = f"tactistruct_npz_{split}"

    out_dir = os.path.join(category_dir, output_folder_name)

    if not os.path.isdir(obj_dir):
        print(f"[WARN] skip missing dir: {obj_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)

    obj_files = sorted([
        f for f in os.listdir(obj_dir)
        if f.lower().endswith(".obj")
    ])[:max_objects]

    print(f"[INFO] {split}: found {len(obj_files)} obj files in {obj_dir}")

    for name in obj_files:
        obj_path = os.path.join(obj_dir, name)
        out_path = os.path.join(out_dir, os.path.splitext(name)[0] + ".npz")

        if os.path.exists(out_path):
            print("[SKIP exists]", out_path)
            continue

        try:
            process_single_obj_to_merged_npz(
                obj_path=obj_path,
                out_path=out_path,
                num_tactile_samples=num_tactile_samples,
            )
        except Exception as e:
            print("[FAILED]", obj_path)
            print("Error:", e)


# =========================================================
# 18. process all categories
# =========================================================
def process_all_categories(
    root_dir,
    split="train",
    max_objects_per_category=80,
    num_tactile_samples=10,
    category_names=None,
):
    subdirs = sorted([
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    if category_names is not None:
        category_names = set(category_names)
        subdirs = [d for d in subdirs if os.path.basename(d) in category_names]

    if not subdirs:
        print("[WARN] no category folders found under:", root_dir)
        return

    print(f"[INFO] found {len(subdirs)} category folders under root.")

    for category_dir in subdirs:
        category_name = os.path.basename(category_dir)
        print(f"\n########## Processing category: {category_name} ##########")

        process_split(
            category_dir=category_dir,
            split=split,
            max_objects=max_objects_per_category,
            num_tactile_samples=num_tactile_samples,
            output_folder_name=f"tactistruct_npz_{split}",
        )


# =========================================================
# 19. main
# =========================================================
if __name__ == "__main__":
    root_dir = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40"

    selected_categories = [
        "airplane",
        "bathtub",
        "bed",
        "bench",
        "bookshelf",
        "bottle",
        "bowl",
        "car",
        "cone",
        "cup",
        "curtain",
        "desk",
        "door",
        "dresser",
        "flower_pot",
        "glass_box",
        "guitar",
        "keyboard",
        "lamp",
        "laptop",
        "mantel",
        "monitor",
        "night_stand",
        "person",
        "piano",
        "plant",
        "radio",
        "range_hood",
        "sink",
        "stairs",
        "stool",
        "table",
        "tent",
        "toilet",
        "tv_stand",
        "vase",
        "wardrobe",
        "xbox",
    ]

    process_all_categories(
        root_dir=root_dir,
        split="train",
        max_objects_per_category=80,
        num_tactile_samples=10,
        category_names = selected_categories
    )

    print("\nAll done.")