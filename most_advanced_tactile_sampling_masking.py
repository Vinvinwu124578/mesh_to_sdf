# import numpy as np
# import trimesh
# import open3d as o3d


# def scale_to_unit_sphere(mesh):
#     if isinstance(mesh, trimesh.Scene):
#         mesh = mesh.dump().sum()

#     vertices = mesh.vertices - mesh.bounding_box.centroid
#     distances = np.linalg.norm(vertices, axis=1)
#     vertices /= np.max(distances)

#     return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


# class SurfacePointCloud:
#     def __init__(self, mesh, points, normals=None):
#         self.mesh = mesh
#         self.points = points
#         self.normals = normals


# def sample_from_mesh(mesh, sample_point_count=800000, calculate_normals=True):

#     if calculate_normals:
#         points, face_indices = mesh.sample(sample_point_count, return_index=True)
#         normals = mesh.face_normals[face_indices]
#     else:
#         points = mesh.sample(sample_point_count, return_index=False)
#         normals = None

#     return SurfacePointCloud(mesh, points=points, normals=normals)


# def bbox_diag(points):
#     mn = points.min(axis=0)
#     mx = points.max(axis=0)
#     return float(np.linalg.norm(mx - mn))


# def extract_finger_patch(points, normals, center_idx, radius,
#                          normal_angle_deg=28.0, min_points=150):

#     c = points[center_idx]
#     nc = normals[center_idx]

#     d = np.linalg.norm(points - c[None, :], axis=1)
#     spatial_mask = d <= radius

#     # v = points - c[None, :]
#     # height = np.abs(v @ nc)
#     v = points - c
#     height = v @ nc

#     idx = np.where(mask)[0]
#     heights = height[idx]


#     layer_center = np.median(heights)
#     layer_thickness = radius * 0.05

#     layer_mask = np.abs(height - layer_center) < layer_thickness
#     # thickness = radius * 0.15
#     # height_mask = height <= thickness

#     cos_th = np.cos(np.deg2rad(normal_angle_deg))
#     dot = (normals @ nc)
#     normal_mask = dot >= cos_th

#     mask = spatial_mask & normal_mask & layer_mask

#     if int(mask.sum()) < int(min_points):
#         return None

#     return mask


# def tactile_sample_5_random_fingers_no_constraints(
#     spc: SurfacePointCloud,
#     points_per_finger=3000,
#     patch_radius_ratio=0.06,
#     normal_angle_deg=28.0,
#     min_points=150,
#     max_trials_per_finger=200,
#     rng=None,
# ):

#     assert spc.normals is not None, "need normals"

#     points = spc.points.astype(np.float32)
#     normals = spc.normals.astype(np.float32)

#     diag = bbox_diag(points)

#     if rng is None:
#         rng = np.random.default_rng()

#     radius = patch_radius_ratio * diag

#     all_pts = []
#     all_fid = []

#     light_dir = np.array([0,0,1], dtype=np.float32)

#     light_dirs = np.array([
#     [1,0,0],[-1,0,0],
#     [0,1,0],[0,-1,0],
#     [0,0,1],[0,0,-1]
# ], dtype=np.float32)

#     visible_mask = (normals @ light_dir) > 0

#     visible_indices = np.where(visible_mask)[0]

#     for fid in range(5):

#         success = False

#         for _ in range(max_trials_per_finger):

#             # center_idx = int(rng.integers(0, len(points)))
#             center_idx = int(rng.choice(visible_indices))

#             # mask = extract_finger_patch(
#             #     points,
#             #     normals,
#             #     center_idx,
#             #     radius,
#             #     normal_angle_deg,
#             #     min_points
#             # )

#             # if mask is None:
#             #     continue

#             c = points[center_idx]
#             nc = normals[center_idx]

#             d = np.linalg.norm(points - c[None, :], axis=1)
#             spatial_mask = d <= radius

#             cos_th = np.cos(np.deg2rad(normal_angle_deg))
#             dot = normals @ nc
#             normal_mask = dot >= cos_th

#             mask = spatial_mask & normal_mask

#             spatial_count = int(spatial_mask.sum())
#             normal_count = int(mask.sum())

#             if normal_count < min_points:

#                 print(f"[Finger {fid}] center_idx={center_idx}")
#                 print(f"  spatial points: {spatial_count}")
#                 print(f"  after normal filter: {normal_count}")
#                 print(f"  required min_points: {min_points}")
#                 print(f"  radius={radius:.4f}  normal_angle={normal_angle_deg}")

#                 continue



#             idx = np.where(mask)[0]

#             # if len(idx) >= points_per_finger:
#             #     choose = rng.choice(idx, size=points_per_finger, replace=False)
#             # else:
#             #     choose = rng.choice(idx, size=points_per_finger, replace=True)
#             choose = idx

#             all_pts.append(points[choose])
#             # all_fid.append(np.full(points_per_finger, fid))
#             all_fid.append(np.full(len(choose), fid))

#             success = True
#             break

#         print(f"[INFO] finger {fid} sampled {len(choose)} points")

#         if not success:
#             raise RuntimeError("Failed to sample finger")

#     sampled_points = np.vstack(all_pts)
#     finger_id = np.concatenate(all_fid)

#     return sampled_points, finger_id


# # -------------------------
# # Open3D visualization
# # -------------------------
# # def visualize(points, finger_points, finger_ids):

# #     pcd = o3d.geometry.PointCloud()
# #     pcd.points = o3d.utility.Vector3dVector(points)

# #     base_color = np.tile([0.7,0.7,0.7], (len(points),1))
# #     pcd.colors = o3d.utility.Vector3dVector(base_color)

# #     geometries = [pcd]

# #     colors = [
# #         [1,0,0],
# #         [0,0,1],
# #         [0,1,0],
# #         [1,0.5,0],
# #         [1,0,1]
# #     ]

# #     for fid in range(5):

# #         pts = finger_points[finger_ids == fid]

# #         p = o3d.geometry.PointCloud()
# #         p.points = o3d.utility.Vector3dVector(pts)

# #         color = np.tile(colors[fid], (len(pts),1))
# #         p.colors = o3d.utility.Vector3dVector(color)

# #         geometries.append(p)

# #     o3d.visualization.draw_geometries(geometries)

# def visualize(points, finger_points, finger_ids):

#     geometries = []

#     # -----------------------------
#     # 稀疏物体点云
#     # -----------------------------
#     idx = np.random.choice(len(points), 50000, replace=False)
#     object_points = points[idx]

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(object_points)

#     colors = np.tile([0.8,0.8,0.8], (len(object_points),1))
#     pcd.colors = o3d.utility.Vector3dVector(colors)

#     geometries.append(pcd)

#     # -----------------------------
#     # tactile points
#     # -----------------------------
#     finger_colors = [
#         [1,0,0],
#         [0,0,1],
#         [0,1,0],
#         [1,0.5,0],
#         [1,0,1]
#     ]

#     for fid in range(5):

#         pts = finger_points[finger_ids == fid]

#         print(f"[VIS] finger {fid} points={len(pts)}")

#         if len(pts) == 0:
#             continue

#         p = o3d.geometry.PointCloud()
#         p.points = o3d.utility.Vector3dVector(pts)

#         color = np.tile(finger_colors[fid], (len(pts),1))
#         p.colors = o3d.utility.Vector3dVector(color)

#         geometries.append(p)

#     # -----------------------------
#     # 显示
#     # -----------------------------
#     o3d.visualization.draw_geometries(
#         geometries,
#         window_name="Tactile Sampling",
#         point_show_normal=False
#     )


# # -------------------------
# # main process
# # -------------------------
# def process_single_obj(obj_path,
#                        sample_point_count=800000,
#                        points_per_finger=3000,
#                        rounds=5,
#                        start_ratio=0.12,
#                        end_ratio=0.03):

#     mesh = trimesh.load(obj_path, force="mesh")
#     mesh.process(validate=True)

#     mesh = scale_to_unit_sphere(mesh)

#     spc = sample_from_mesh(mesh, sample_point_count)

#     radius_schedule = np.linspace(start_ratio, end_ratio, rounds)

#     for rid, radius_ratio in enumerate(radius_schedule):

#         print(f"Round {rid} radius_ratio={radius_ratio:.4f}")

#         finger_points, finger_id = tactile_sample_5_random_fingers_no_constraints(
#             spc,
#             points_per_finger=points_per_finger,
#             patch_radius_ratio=float(radius_ratio),
#         )

#         visualize(spc.points, finger_points, finger_id)


# if __name__ == "__main__":

#     # OBJ_PATH = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/sofa/train_obj/sofa_0002.obj"
#     OBJ_PATH = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/airplane/train_obj/airplane_0002.obj"

#     process_single_obj(
#         OBJ_PATH,
#         sample_point_count=800000,
#         points_per_finger=3000,
#         rounds=5,
#         start_ratio=0.12,
#         end_ratio=0.03
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
#     vertices /= np.max(distances)

#     return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


# # -------------------------------------------------
# # point cloud container
# # -------------------------------------------------
# class SurfacePointCloud:
#     def __init__(self, mesh, points, normals=None):
#         self.mesh = mesh
#         self.points = points
#         self.normals = normals


# # -------------------------------------------------
# # sample mesh surface
# # -------------------------------------------------
# def sample_from_mesh(mesh, sample_point_count=200000, calculate_normals=True):

#     if calculate_normals:
#         points, face_indices = mesh.sample(sample_point_count, return_index=True)
#         normals = mesh.face_normals[face_indices]
#     else:
#         points = mesh.sample(sample_point_count)
#         normals = None

#     return SurfacePointCloud(mesh, points=points, normals=normals)


# # -------------------------------------------------
# # bounding box diagonal
# # -------------------------------------------------
# def bbox_diag(points):
#     mn = points.min(axis=0)
#     mx = points.max(axis=0)
#     return float(np.linalg.norm(mx - mn))


# # -------------------------------------------------
# # Open3D ray casting
# # -------------------------------------------------
# def get_outer_surface_indices(mesh, points, num_rays=10000):

#     print("[INFO] Running Open3D ray casting")

#     mesh_o3d = o3d.t.geometry.TriangleMesh.from_legacy(
#         o3d.geometry.TriangleMesh(
#             o3d.utility.Vector3dVector(mesh.vertices),
#             o3d.utility.Vector3iVector(mesh.faces)
#         )
#     )

#     scene = o3d.t.geometry.RaycastingScene()
#     scene.add_triangles(mesh_o3d)

#     # random directions
#     dirs = np.random.normal(size=(num_rays,3))
#     dirs /= np.linalg.norm(dirs,axis=1)[:,None]

#     origins = dirs * 3.0
#     directions = -dirs

#     rays = np.concatenate([origins, directions], axis=1)

#     rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

#     ans = scene.cast_rays(rays)

#     t_hit = ans["t_hit"].numpy()

#     hit_mask = np.isfinite(t_hit)

#     hit_points = origins[hit_mask] + directions[hit_mask] * t_hit[hit_mask][:,None]

#     print("[INFO] ray hits:", len(hit_points))

#     tree = cKDTree(points)

#     _, ids = tree.query(hit_points, k=1)

#     ids = np.unique(ids)

#     print("[INFO] outer surface candidates:", len(ids))

#     return ids


# # -------------------------------------------------
# # tactile sampling
# # -------------------------------------------------
# def tactile_sample_5_random_fingers_no_constraints(
#     spc: SurfacePointCloud,
#     outer_surface_ids,
#     points_per_finger=3000,
#     patch_radius_ratio=0.06,
#     normal_angle_deg=28.0,
#     min_points=150,
#     max_trials_per_finger=200,
# ):

#     points = spc.points.astype(np.float32)
#     normals = spc.normals.astype(np.float32)

#     diag = bbox_diag(points)

#     rng = np.random.default_rng()

#     radius = patch_radius_ratio * diag

#     all_pts = []
#     all_fid = []

#     for fid in range(5):

#         success = False

#         for _ in range(max_trials_per_finger):

#             center_idx = int(rng.choice(outer_surface_ids))

#             c = points[center_idx]
#             nc = normals[center_idx]

#             d = np.linalg.norm(points - c[None, :], axis=1)
#             spatial_mask = d <= radius

#             cos_th = np.cos(np.deg2rad(normal_angle_deg))
#             dot = normals @ nc

#             normal_mask = dot >= cos_th

#             mask = spatial_mask & normal_mask

#             if mask.sum() < min_points:
#                 continue

#             idx = np.where(mask)[0]

#             choose = idx

#             all_pts.append(points[choose])
#             all_fid.append(np.full(len(choose), fid))

#             success = True
#             break

#         if not success:
#             raise RuntimeError("Failed to sample finger")

#         print(f"[INFO] finger {fid} sampled {len(choose)} points")

#     sampled_points = np.vstack(all_pts)
#     finger_id = np.concatenate(all_fid)

#     return sampled_points, finger_id


# # -------------------------------------------------
# # visualization
# # -------------------------------------------------
# def visualize(points, finger_points, finger_ids):

#     geometries = []

#     idx = np.random.choice(len(points), 50000, replace=False)
#     object_points = points[idx]

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(object_points)

#     colors = np.tile([0.8,0.8,0.8], (len(object_points),1))
#     pcd.colors = o3d.utility.Vector3dVector(colors)

#     geometries.append(pcd)

#     finger_colors = [
#         [1,0,0],
#         [0,0,1],
#         [0,1,0],
#         [1,0.5,0],
#         [1,0,1]
#     ]

#     for fid in range(5):

#         pts = finger_points[finger_ids == fid]

#         print(f"[VIS] finger {fid} points={len(pts)}")

#         if len(pts) == 0:
#             continue

#         p = o3d.geometry.PointCloud()
#         p.points = o3d.utility.Vector3dVector(pts)

#         color = np.tile(finger_colors[fid], (len(pts),1))
#         p.colors = o3d.utility.Vector3dVector(color)

#         geometries.append(p)

#     o3d.visualization.draw_geometries(
#         geometries,
#         window_name="Tactile Sampling",
#         point_show_normal=False
#     )


# # -------------------------------------------------
# # main process
# # -------------------------------------------------
# def process_single_obj(obj_path,
#                        sample_point_count=200000,
#                        points_per_finger=3000,
#                        rounds=5,
#                        start_ratio=0.12,
#                        end_ratio=0.03):

#     mesh = trimesh.load(obj_path, force="mesh")
#     mesh.process(validate=True)

#     mesh = scale_to_unit_sphere(mesh)

#     spc = sample_from_mesh(mesh, sample_point_count)

#     outer_surface_ids = get_outer_surface_indices(mesh, spc.points)

#     radius_schedule = np.linspace(start_ratio, end_ratio, rounds)

#     for rid, radius_ratio in enumerate(radius_schedule):

#         print(f"Round {rid} radius_ratio={radius_ratio:.4f}")

#         finger_points, finger_id = tactile_sample_5_random_fingers_no_constraints(
#             spc,
#             outer_surface_ids,
#             points_per_finger=points_per_finger,
#             patch_radius_ratio=float(radius_ratio),
#         )

#         visualize(spc.points, finger_points, finger_id)


# # -------------------------------------------------
# # main
# # -------------------------------------------------
# if __name__ == "__main__":

#     OBJ_PATH = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/sofa/train_obj/sofa_0002.obj"

#     process_single_obj(
#         OBJ_PATH,
#         sample_point_count=200000,
#         points_per_finger=3000,
#         rounds=5,
#         start_ratio=0.12,
#         end_ratio=0.03
#     )















import numpy as np
import trimesh
import open3d as o3d
from scipy.spatial import cKDTree


# -------------------------------------------------
# normalize mesh
# -------------------------------------------------
def scale_to_unit_sphere(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


# -------------------------------------------------
# point cloud container
# -------------------------------------------------
class SurfacePointCloud:
    def __init__(self, mesh, points, normals=None):
        self.mesh = mesh
        self.points = points
        self.normals = normals


# -------------------------------------------------
# sample mesh surface
# -------------------------------------------------
def sample_from_mesh(mesh, sample_point_count=200000, calculate_normals=True):
    if calculate_normals:
        points, face_indices = mesh.sample(sample_point_count, return_index=True)
        normals = mesh.face_normals[face_indices]
    else:
        points = mesh.sample(sample_point_count)
        normals = None

    return SurfacePointCloud(mesh, points=points, normals=normals)


# -------------------------------------------------
# bounding box diagonal
# -------------------------------------------------
def bbox_diag(points):
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    return float(np.linalg.norm(mx - mn))


# -------------------------------------------------
# Open3D ray casting: get outer surface candidate indices
# -------------------------------------------------
def get_outer_surface_indices(mesh, points, num_rays=10000):
    print("[INFO] Running Open3D ray casting...")

    legacy_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices.astype(np.float64)),
        o3d.utility.Vector3iVector(mesh.faces.astype(np.int32))
    )

    mesh_o3d = o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh)

    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_o3d)

    # random directions on sphere
    dirs = np.random.normal(size=(num_rays, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    # origins outside normalized unit sphere
    origins = dirs * 3.0
    directions = -dirs

    rays = np.concatenate([origins, directions], axis=1).astype(np.float32)
    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

    ans = scene.cast_rays(rays)
    t_hit = ans["t_hit"].numpy()

    hit_mask = np.isfinite(t_hit)
    hit_points = origins[hit_mask] + directions[hit_mask] * t_hit[hit_mask][:, None]

    print("[INFO] ray hits:", len(hit_points))
    if len(hit_points) == 0:
        raise RuntimeError("Ray casting found no hits.")

    tree = cKDTree(points)
    _, ids = tree.query(hit_points, k=1)
    ids = np.unique(ids)

    print("[INFO] outer surface candidates:", len(ids))
    return ids


# -------------------------------------------------
# tangent plane circular patch
# -------------------------------------------------
def extract_tangent_plane_patch(
    points,
    normals,
    center_idx,
    radius,
    thickness,
    normal_angle_deg=28.0,
    min_points=150,
):
    c = points[center_idx]
    nc = normals[center_idx]

    # make sure nc is unit-length
    nc_norm = np.linalg.norm(nc)
    if nc_norm < 1e-12:
        return None
    nc = nc / nc_norm

    # vector from center to all points
    v = points - c[None, :]

    # signed distance along normal direction
    height = v @ nc

    # projection to tangent plane
    v_plane = v - height[:, None] * nc[None, :]
    plane_dist = np.linalg.norm(v_plane, axis=1)

    # circular patch on tangent plane
    plane_mask = plane_dist <= radius

    # thin slab around tangent plane
    thickness_mask = np.abs(height) <= thickness

    # optional normal consistency
    cos_th = np.cos(np.deg2rad(normal_angle_deg))
    dot = np.sum(normals * nc[None, :], axis=1)
    normal_mask = dot >= cos_th

    mask = plane_mask & thickness_mask & normal_mask

    if int(mask.sum()) < int(min_points):
        return None

    return mask


# -------------------------------------------------
# tactile sampling with tangent plane circular patches
# -------------------------------------------------
def tactile_sample_5_random_fingers_tangent_plane(
    spc: SurfacePointCloud,
    outer_surface_ids,
    points_per_finger=3000,
    patch_radius_ratio=0.06,
    thickness_ratio=0.01,
    normal_angle_deg=28.0,
    min_points=150,
    max_trials_per_finger=200,
    rng=None,
):
    assert spc.normals is not None, "Normals are required."

    points = spc.points.astype(np.float32)
    normals = spc.normals.astype(np.float32)

    # normalize normals
    nrm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(nrm, 1e-12, None)

    if rng is None:
        rng = np.random.default_rng()

    diag = bbox_diag(points)
    radius = patch_radius_ratio * diag
    thickness = thickness_ratio * diag

    all_pts = []
    all_fid = []
    center_ids = []

    print(f"[INFO] tangent plane patch radius    = {radius:.6f}")
    print(f"[INFO] tangent plane patch thickness = {thickness:.6f}")

    for fid in range(5):
        success = False

        for _ in range(max_trials_per_finger):
            center_idx = int(rng.choice(outer_surface_ids))

            mask = extract_tangent_plane_patch(
                points=points,
                normals=normals,
                center_idx=center_idx,
                radius=radius,
                thickness=thickness,
                normal_angle_deg=normal_angle_deg,
                min_points=min_points,
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
            center_ids.append(center_idx)

            success = True
            break

        if not success:
            raise RuntimeError(
                f"Failed to sample finger {fid}. "
                f"Try increasing patch_radius_ratio or thickness_ratio."
            )

        print(f"[INFO] finger {fid} sampled {points_per_finger} points")

    sampled_points = np.vstack(all_pts)
    finger_id = np.concatenate(all_fid)

    return sampled_points, finger_id, center_ids


# -------------------------------------------------
# visualization
# -------------------------------------------------
def visualize(points, finger_points, finger_ids, center_ids=None):
    geometries = []

    # sparse object points
    bg_count = min(50000, len(points))
    idx = np.random.choice(len(points), bg_count, replace=False)
    object_points = points[idx]

    bg = o3d.geometry.PointCloud()
    bg.points = o3d.utility.Vector3dVector(object_points.astype(np.float64))
    bg.colors = o3d.utility.Vector3dVector(
        np.tile([0.8, 0.8, 0.8], (len(object_points), 1)).astype(np.float64)
    )
    geometries.append(bg)

    finger_colors = [
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0.5, 0],
        [1, 0, 1],
    ]

    for fid in range(5):
        pts = finger_points[finger_ids == fid]
        print(f"[VIS] finger {fid} points = {len(pts)}")

        if len(pts) == 0:
            continue

        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        p.colors = o3d.utility.Vector3dVector(
            np.tile(finger_colors[fid], (len(pts), 1)).astype(np.float64)
        )
        geometries.append(p)

    # center points as black spheres
    if center_ids is not None and len(center_ids) > 0:
        centers = points[np.array(center_ids, dtype=int)]
        for c in centers:
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            s.paint_uniform_color([0, 0, 0])
            s.translate(c.astype(np.float64))
            geometries.append(s)

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Tactile Sampling - Tangent Plane Circular Patch",
        point_show_normal=False
    )


# -------------------------------------------------
# main process
# -------------------------------------------------
def process_single_obj(
    obj_path,
    sample_point_count=200000,
    points_per_finger=3000,
    rounds=5,
    start_ratio=0.12,
    end_ratio=0.03,
    thickness_ratio=0.01,
    normal_angle_deg=28.0,
    num_rays=10000,
):
    mesh = trimesh.load(obj_path, force="mesh")
    mesh.process(validate=True)

    mesh = scale_to_unit_sphere(mesh)

    spc = sample_from_mesh(mesh, sample_point_count=sample_point_count, calculate_normals=True)

    # only choose centers from outer surface
    outer_surface_ids = get_outer_surface_indices(mesh, spc.points, num_rays=num_rays)

    radius_schedule = np.linspace(start_ratio, end_ratio, rounds)

    for rid, radius_ratio in enumerate(radius_schedule):
        print(f"\n[ROUND {rid}] patch_radius_ratio = {radius_ratio:.4f}")

        finger_points, finger_id, center_ids = tactile_sample_5_random_fingers_tangent_plane(
            spc=spc,
            outer_surface_ids=outer_surface_ids,
            points_per_finger=points_per_finger,
            patch_radius_ratio=float(radius_ratio),
            thickness_ratio=thickness_ratio,
            normal_angle_deg=normal_angle_deg,
        )

        visualize(spc.points, finger_points, finger_id, center_ids=center_ids)


# -------------------------------------------------
# main
# -------------------------------------------------
if __name__ == "__main__":
    # OBJ_PATH = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/sofa/train_obj/sofa_0002.obj"
    OBJ_PATH = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/chair/train_obj/chair_0002.obj"

    process_single_obj(
        obj_path=OBJ_PATH,
        sample_point_count=200000,
        points_per_finger=3000,
        rounds=5,
        start_ratio=0.12,
        end_ratio=0.03,
        thickness_ratio=0.01,   # 可调：0.005 ~ 0.02
        normal_angle_deg=28.0,
        num_rays=10000,
    )