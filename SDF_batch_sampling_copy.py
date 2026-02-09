# import os
# import numpy as np
# import trimesh
# from mesh_to_sdf import sample_sdf_near_surface
# from concurrent.futures import ProcessPoolExecutor, as_completed


# # =========================================================
# # 指定要处理的 ModelNet40 类别（与截图一致，可自行删减）
# # =========================================================
# TARGET_CATEGORIES = [
#     "guitar",
#     "keyboard",
#     "lamp",
#     "laptop",
#     "mantel",
#     "monitor",
#     "night_stand",
#     "person",
#     "piano",
#     "plant",
#     "radio",
#     "range_hood",
#     "sink",
#     "sofa",
#     "stairs",
#     "stool",
#     "table",
#     "tent",
#     "toilet",
#     "tv_stand",
#     "vase",
#     "wardrobe",
#     "xbox"
# ]



# def process_split(category_dir, split, num_points=250000):
#     """
#     处理单个 split（train / test）
#     """
#     obj_dir = os.path.join(category_dir, f"{split}_obj")
#     out_dir = os.path.join(category_dir, f"sdf_npz_{split}")

#     if not os.path.isdir(obj_dir):
#         return

#     os.makedirs(out_dir, exist_ok=True)

#     for name in os.listdir(obj_dir):
#         if not name.lower().endswith(".obj"):
#             continue

#         obj_path = os.path.join(obj_dir, name)
#         out_path = os.path.join(out_dir, name.replace(".obj", ".npz"))

#         if os.path.exists(out_path):
#             print("Skip (exists):", out_path)
#             continue

#         try:
#             # 关键：process=False，减少 trimesh 内部渲染依赖
#             mesh = trimesh.load(obj_path, force="mesh", process=False)

#             points, sdf = sample_sdf_near_surface(
#                 mesh,
#                 number_of_points=num_points,
#                 scan_count=0
#             )

#             np.savez(
#                 out_path,
#                 points=points.astype(np.float32),
#                 sdf=sdf.astype(np.float32)
#             )

#             print("Saved:", out_path)

#         except Exception as e:
#             print("Failed:", obj_path, "Error:", e)


# def process_category(root_dir, category, num_points):
#     """
#     单个类别处理函数（用于多进程）
#     """
#     category_dir = os.path.join(root_dir, category)

#     if not os.path.isdir(category_dir):
#         print(f"[Skip] Not a directory: {category}")
#         return

#     # 如果已存在 sdf_npz_*，整个类别跳过
#     if (
#         os.path.isdir(os.path.join(category_dir, "sdf_npz_train"))
#         or os.path.isdir(os.path.join(category_dir, "sdf_npz_test"))
#     ):
#         print(f"[Skip] {category} already has sdf_npz folders")
#         return

#     print(f"\n[Start] Processing category: {category}")

#     process_split(category_dir, "train", num_points)
#     process_split(category_dir, "test", num_points)

#     print(f"[Done] {category}")


# def process_modelnet40_multiprocess(root_dir, num_points=235000, max_workers=4):
#     """
#     多进程入口（Windows 安全）
#     """
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         futures = []

#         for category in TARGET_CATEGORIES:
#             futures.append(
#                 executor.submit(process_category, root_dir, category, num_points)
#             )

#         for f in as_completed(futures):
#             # 如果子进程抛异常，这里会直接报出来
#             f.result()


# if __name__ == "__main__":
#     root_dir = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40"

#     process_modelnet40_multiprocess(
#         root_dir=root_dir,
#         num_points=235000,
#         max_workers=4   # Windows 建议 2~4
#     )

#     print("\nAll SDF sampling done.")






TARGET_CATEGORIES = [
    "guitar", 
    "piano", "plant", "radio", "range_hood", "sink", "sofa",
    "stairs", "stool", "table", "tent", "toilet", "tv_stand",
    "vase", "wardrobe", "xbox",
    "keyboard", "lamp",
    "laptop", "mantel", "monitor", "night_stand", "person",
    
    "chair","car","bowl","bottle",
    "cone", "cup", "curtain", "desk", "door", "dresser",
    "flower_pot", "glass_box",
]






import os
import trimesh
import numpy as np
from mesh_to_sdf import sample_sdf_near_surface
from concurrent.futures import ProcessPoolExecutor, as_completed

def _process_one_obj(obj_path, out_path, num_points):
    """子进程执行：单个 obj -> npz"""
    if os.path.exists(out_path):
        return ("skip", out_path, None)

    try:
        mesh = trimesh.load(obj_path, force='mesh')
        points, sdf = sample_sdf_near_surface(mesh, number_of_points=num_points)

        np.savez(
            out_path,
            points=points.astype(np.float32),
            sdf=sdf.astype(np.float32)
        )
        return ("ok", out_path, None)
    except Exception as e:
        return ("fail", obj_path, str(e))


def process_split(category_dir, split, num_points=235000, max_files=80, max_workers=None):
    """
    category_dir: ModelNet40/airplane
    split: 'train' or 'test'
    max_workers: 进程数，None=默认（一般等于CPU核心数）
    """
    obj_dir = os.path.join(category_dir, f"{split}_obj")
    out_dir = os.path.join(category_dir, f"sdf_npz_{split}_same_as_ori_paper")

    if not os.path.isdir(obj_dir):
        return

    os.makedirs(out_dir, exist_ok=True)

    # 只取前 max_files 个 obj（排序保证可复现）
    obj_files = sorted([f for f in os.listdir(obj_dir) if f.lower().endswith(".obj")])[:max_files]

    tasks = []
    for name in obj_files:
        obj_path = os.path.join(obj_dir, name)
        out_path = os.path.join(out_dir, name.replace(".obj", ".npz"))
        tasks.append((obj_path, out_path, num_points))

    if not tasks:
        return

    print(f"  {split}: submit {len(tasks)} files, workers={max_workers}")

    # 多进程并行
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_one_obj, *t) for t in tasks]

        for fut in as_completed(futures):
            status, path, err = fut.result()
            if status == "ok":
                print("Saved:", path)
            elif status == "skip":
                print("Skip (exists):", path)
            else:
                print("Failed:", path, "Error:", err)


# def process_modelnet40(root_dir, num_points=235000, max_files=80, max_workers=None):
#     for category in os.listdir(root_dir):
#         category_dir = os.path.join(root_dir, category)
#         if not os.path.isdir(category_dir):
#             continue

#         print(f"\nProcessing category: {category}")
#         process_split(category_dir, "train", num_points, max_files=max_files, max_workers=max_workers)
#         process_split(category_dir, "test", num_points, max_files=max_files, max_workers=max_workers)

# def process_modelnet40(root_dir, num_points=235000, max_files=80, max_workers=None):
#     for category in sorted(os.listdir(root_dir)):
#         # 👉 只处理 target_categories
#         if category not in TARGET_CATEGORIES:
#             continue

#         category_dir = os.path.join(root_dir, category)
#         if not os.path.isdir(category_dir):
#             continue

#         print(f"\nProcessing category: {category}")
#         process_split(
#             category_dir,
#             "train",
#             num_points=num_points,
#             max_files=max_files,
#             max_workers=max_workers
#         )
#         process_split(
#             category_dir,
#             "test",
#             num_points=num_points,
#             max_files=max_files,
#             max_workers=max_workers
#         )
def process_modelnet40(root_dir, num_points=235000, max_files=80, max_workers=None):
    for category in TARGET_CATEGORIES:  # 👈 顺序完全由你控制
        category_dir = os.path.join(root_dir, category)

        if not os.path.isdir(category_dir):
            print(f"[Skip] Category folder not found: {category}")
            continue

        print(f"\nProcessing category: {category}")

        process_split(
            category_dir,
            "train",
            num_points=num_points,
            max_files=max_files,
            max_workers=max_workers
        )
        # process_split(
        #     category_dir,
        #     "test",
        #     num_points=num_points,
        #     max_files=max_files,
        #     max_workers=max_workers
        # )



if __name__ == "__main__":
    root_dir = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40"
    # max_workers 建议先试：CPU核心数-1，比如 7/11/15；或直接 None
    process_modelnet40(root_dir, num_points=235000, max_files=80, max_workers=5)
    print("\nAll SDF sampling done.")
