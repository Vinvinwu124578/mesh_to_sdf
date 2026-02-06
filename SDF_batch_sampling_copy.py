import os
import numpy as np
import trimesh
from mesh_to_sdf import sample_sdf_near_surface
from concurrent.futures import ProcessPoolExecutor, as_completed


# =========================================================
# 指定要处理的 ModelNet40 类别（与截图一致，可自行删减）
# =========================================================
TARGET_CATEGORIES = [
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
    "sofa",
    "stairs",
    "stool",
    "table",
    "tent",
    "toilet",
    "tv_stand",
    "vase",
    "wardrobe",
    "xbox"
]



def process_split(category_dir, split, num_points=250000):
    """
    处理单个 split（train / test）
    """
    obj_dir = os.path.join(category_dir, f"{split}_obj")
    out_dir = os.path.join(category_dir, f"sdf_npz_{split}")

    if not os.path.isdir(obj_dir):
        return

    os.makedirs(out_dir, exist_ok=True)

    for name in os.listdir(obj_dir):
        if not name.lower().endswith(".obj"):
            continue

        obj_path = os.path.join(obj_dir, name)
        out_path = os.path.join(out_dir, name.replace(".obj", ".npz"))

        if os.path.exists(out_path):
            print("Skip (exists):", out_path)
            continue

        try:
            # 关键：process=False，减少 trimesh 内部渲染依赖
            mesh = trimesh.load(obj_path, force="mesh", process=False)

            points, sdf = sample_sdf_near_surface(
                mesh,
                number_of_points=num_points,
                scan_count=0
            )

            np.savez(
                out_path,
                points=points.astype(np.float32),
                sdf=sdf.astype(np.float32)
            )

            print("Saved:", out_path)

        except Exception as e:
            print("Failed:", obj_path, "Error:", e)


def process_category(root_dir, category, num_points):
    """
    单个类别处理函数（用于多进程）
    """
    category_dir = os.path.join(root_dir, category)

    if not os.path.isdir(category_dir):
        print(f"[Skip] Not a directory: {category}")
        return

    # 如果已存在 sdf_npz_*，整个类别跳过
    if (
        os.path.isdir(os.path.join(category_dir, "sdf_npz_train"))
        or os.path.isdir(os.path.join(category_dir, "sdf_npz_test"))
    ):
        print(f"[Skip] {category} already has sdf_npz folders")
        return

    print(f"\n[Start] Processing category: {category}")

    process_split(category_dir, "train", num_points)
    process_split(category_dir, "test", num_points)

    print(f"[Done] {category}")


def process_modelnet40_multiprocess(root_dir, num_points=250000, max_workers=4):
    """
    多进程入口（Windows 安全）
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for category in TARGET_CATEGORIES:
            futures.append(
                executor.submit(process_category, root_dir, category, num_points)
            )

        for f in as_completed(futures):
            # 如果子进程抛异常，这里会直接报出来
            f.result()


if __name__ == "__main__":
    root_dir = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40"

    process_modelnet40_multiprocess(
        root_dir=root_dir,
        num_points=250000,
        max_workers=4   # Windows 建议 2~4
    )

    print("\nAll SDF sampling done.")
