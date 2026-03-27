# batch_tactile_sampling_modelnet40.py
import os
import numpy as np
import trimesh

def scale_to_unit_sphere(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

class SurfacePointCloud:
    def __init__(self, mesh, points, normals=None, scans=None):
        self.mesh = mesh
        self.points = points
        self.normals = normals
        self.scans = scans


def sample_from_mesh(mesh, sample_point_count=800_000, calculate_normals=True):
    """
    不固定 seed：同一物体多次运行会得到不同的表面点云（如果你希望可复现，见下方注释）
    """
    if calculate_normals:
        points, face_indices = mesh.sample(sample_point_count, return_index=True)
        normals = mesh.face_normals[face_indices]
    else:
        points = mesh.sample(sample_point_count, return_index=False)
        normals = None
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
    rng=None,
):
    """
    5 个手指：每个手指独立随机找中心 + 圆形 patch（半径）+ 法向过滤
    手指之间不做几何限制：允许重叠、允许中心很近
    """
    assert spc.normals is not None, "need normals"

    points = spc.points.astype(np.float32)
    normals = spc.normals.astype(np.float32)
    diag = bbox_diag(points)
    if diag <= 0:
        raise ValueError("degenerate point cloud bbox")

    if rng is None:
        rng = np.random.default_rng()

    radius = patch_radius_ratio * diag

    all_pts = []
    all_fid = []
    all_sdf = []
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
            all_sdf.append(np.zeros(points_per_finger, dtype=np.float32))
            center_ids.append(center_idx)
            success = True
            break

        if not success:
            raise RuntimeError(
                f"Failed to sample finger {fid}. "
                f"Try increasing patch_radius_ratio (0.08~0.12) or sample_point_count."
            )

    sampled_points = np.vstack(all_pts)
    finger_id = np.concatenate(all_fid)
    sdf = np.concatenate(all_sdf)

    return sampled_points, sdf, finger_id, center_ids


def process_single_obj(obj_path, out_path,
                       sample_point_count=800_000,
                       points_per_finger=3000,
                       patch_radius_ratio=0.06,
                       normal_angle_deg=28.0,
                       min_points=150,
                       seed_mode="object"):
    """
    seed_mode:
      - "free": 每次运行都不同
      - "object": 同一 obj 每次运行一致（推荐做数据集）
      - "fixed": 全部 obj 都用同一个 seed（不推荐）
    """
    # RNG 策略
    if seed_mode == "free":
        rng = np.random.default_rng()
        # trimesh.sample 用 np.random 全局；这里不固定
    elif seed_mode == "fixed":
        rng = np.random.default_rng(0)
        np.random.seed(0)
    elif seed_mode == "object":
        s = (hash(obj_path) & 0xFFFFFFFF)
        rng = np.random.default_rng(s)
        np.random.seed(int(s))
    else:
        raise ValueError("seed_mode must be one of: free/object/fixed")

    # mesh = trimesh.load(obj_path, force="mesh")
    # mesh.process(validate=True)

    mesh = trimesh.load(obj_path, force="mesh")
    mesh.process(validate=True)

    # ⭐ 加入归一化
    mesh = scale_to_unit_sphere(mesh)

    spc = sample_from_mesh(mesh, sample_point_count=sample_point_count, calculate_normals=True)

    points, sdf, finger_id, _ = tactile_sample_5_random_fingers_no_constraints(
        spc,
        points_per_finger=points_per_finger,
        patch_radius_ratio=patch_radius_ratio,
        normal_angle_deg=normal_angle_deg,
        min_points=min_points,
        rng=rng
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        points=points.astype(np.float32),
        sdf=sdf.astype(np.float32),
        finger_id=finger_id.astype(np.int32),
    )


def process_modelnet40(root_dir,
                       sample_point_count=800_000,
                       points_per_finger=3000,
                       patch_radius_ratio=0.06,
                       normal_angle_deg=28.0,
                       min_points=150,
                       seed_mode="object"):
    """
    遍历 root_dir 下所有类别文件夹：
      <category>/train_obj/*.obj  → <category>/tactile_npz_train/*.npz
      <category>/test_obj/*.obj   → <category>/tactile_npz_test/*.npz
    """
    categories = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    print(f"[INFO] Found {len(categories)} categories under: {root_dir}")

    for category in categories:
        category_dir = os.path.join(root_dir, category)
        print(f"\n[CAT] {category}")

        for split in ["train", "test"]:
            obj_dir = os.path.join(category_dir, f"{split}_obj")
            if not os.path.isdir(obj_dir):
                print(f"  [SKIP] No {split}_obj")
                continue

            out_dir = os.path.join(category_dir, f"perfect_tactile_npz_{split}_normalized")
            os.makedirs(out_dir, exist_ok=True)

            obj_names = sorted([n for n in os.listdir(obj_dir) if n.lower().endswith(".obj")])
            print(f"  [{split}] {len(obj_names)} objs")

            for i, name in enumerate(obj_names, 1):
                obj_path = os.path.join(obj_dir, name)
                out_path = os.path.join(out_dir, name[:-4] + ".npz")

                # if os.path.exists(out_path):
                #     print(f"    [SKIP] ({i}/{len(obj_names)}) exists: {name}")
                #     continue

                try:
                    process_single_obj(
                        obj_path, out_path,
                        sample_point_count=sample_point_count,
                        points_per_finger=points_per_finger,
                        patch_radius_ratio=patch_radius_ratio,
                        normal_angle_deg=normal_angle_deg,
                        min_points=min_points,
                        seed_mode=seed_mode
                    )
                    print(f"    [OK]   ({i}/{len(obj_names)}) {name}")
                except Exception as e:
                    print(f"    [FAIL] ({i}/{len(obj_names)}) {name} -> {e}")


if __name__ == "__main__":
    ROOT_DIR = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40"

    process_modelnet40(
        ROOT_DIR,
        sample_point_count=800_000,   # airplane 这类薄结构建议 >= 600k
        points_per_finger=3000,
        patch_radius_ratio=0.06,      # 看不明显可调 0.08~0.12
        normal_angle_deg=28.0,
        min_points=150,
        seed_mode="object"            # 推荐：同一物体每次运行一致；改 "free" 则每次不同
    )

    print("\n[DONE] All categories processed.")
