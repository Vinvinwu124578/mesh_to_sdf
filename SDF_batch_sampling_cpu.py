import os
import trimesh
import numpy as np
from tqdm import tqdm


TARGET_CATEGORIES = [
    "cone", "cup", "curtain", "desk", "door", "dresser",
    "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
    "laptop", "mantel", "monitor", "night_stand", "person",
    "piano", "plant", "radio", "range_hood", "sink", "sofa",
    "stairs", "stool", "table", "tent", "toilet", "tv_stand",
    "vase", "wardrobe", "xbox"
]


# ===============================
# CPU SDF 采样（稳健版）
# ===============================
def sample_sdf_cpu(mesh, num_points=250000):
    # 1. 复制 + 归一化
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.bounding_box.centroid)
    scale = np.max(mesh.bounding_box.extents)
    mesh.apply_scale(1.0 / scale)

    # 2. 采样点
    n_surface = num_points // 2
    n_uniform = num_points - n_surface

    surface_pts, _ = trimesh.sample.sample_surface(mesh, n_surface)
    surface_pts += np.random.normal(scale=0.01, size=surface_pts.shape)

    uniform_pts = np.random.uniform(-1.0, 1.0, size=(n_uniform, 3))
    points = np.vstack([surface_pts, uniform_pts])

    # 3. SDF
    sdf = trimesh.proximity.signed_distance(mesh, points)

    return points.astype(np.float32), sdf.astype(np.float32)


def load_trimesh_safe(obj_path):
    mesh = trimesh.load(obj_path, force="mesh", process=False)

    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            return None
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values()
             if isinstance(g, trimesh.Trimesh)]
        )

    if not isinstance(mesh, trimesh.Trimesh):
        return None

    if mesh.faces.shape[0] < 10:
        return None

    if not mesh.is_watertight:
        mesh = mesh.fill_holes()

    return mesh


def process_split(category_dir, split, num_points):
    obj_dir = os.path.join(category_dir, f"{split}_obj")
    out_dir = os.path.join(category_dir, f"sdf_npz_{split}")
    os.makedirs(out_dir, exist_ok=True)

    obj_files = [f for f in os.listdir(obj_dir) if f.endswith(".obj")]
    print(f"  {split}: {len(obj_files)} meshes")

    for name in tqdm(obj_files):
        obj_path = os.path.join(obj_dir, name)
        out_path = os.path.join(out_dir, name.replace(".obj", ".npz"))

        if os.path.exists(out_path):
            continue

        try:
            mesh = load_trimesh_safe(obj_path)
            if mesh is None:
                continue

            points, sdf = sample_sdf_cpu(mesh, num_points)

            np.savez(out_path, points=points, sdf=sdf)

        except Exception as e:
            print("Failed:", obj_path)
            print("  Error:", e)


def process_modelnet40(root_dir, num_points=250000):
    for cat in TARGET_CATEGORIES:
        cat_dir = os.path.join(root_dir, cat)
        if not os.path.isdir(cat_dir):
            continue

        print(f"\nProcessing category: {cat}")
        process_split(cat_dir, "train", num_points)
        process_split(cat_dir, "test", num_points)


if __name__ == "__main__":
    root_dir = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40"
    process_modelnet40(root_dir)
    print("\nAll SDF sampling done.")
