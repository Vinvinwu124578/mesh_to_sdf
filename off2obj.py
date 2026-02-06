# import trimesh

# m = trimesh.load("model.off")
# m.export("model.obj")

# import os
# import trimesh

# def batch_convert(input_dir, output_dir):
#     # 创建输出文件夹（如果不存在）
#     os.makedirs(output_dir, exist_ok=True)

#     for dirpath, _, filenames in os.walk(input_dir):
#         for name in filenames:
#             if name.lower().endswith(".off"):
#                 off_path = os.path.join(dirpath, name)
                
#                 # 输出文件的路径 —— 全都放进 output_dir
#                 obj_name = name[:-4] + ".obj"
#                 obj_path = os.path.join(output_dir, obj_name)

#                 try:
#                     mesh = trimesh.load(off_path)
#                     mesh.export(obj_path)
#                     print("Converted:", off_path, "->", obj_path)
#                 except Exception as e:
#                     print("Failed:", off_path, "Error:", e)

# if __name__ == "__main__":
#     input_dir  = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/chair/train"
#     output_dir = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/chair/train_obj"

#     batch_convert(input_dir, output_dir)
#     print("Done.")


import os
import trimesh


def convert_split(category_dir, split):
    """
    category_dir: e.g. ModelNet40/chair
    split: 'train' or 'test'
    """
    input_dir = os.path.join(category_dir, split)
    output_dir = os.path.join(category_dir, f"{split}_obj")

    if not os.path.isdir(input_dir):
        return

    os.makedirs(output_dir, exist_ok=True)

    for dirpath, _, filenames in os.walk(input_dir):
        for name in filenames:
            if name.lower().endswith(".off"):
                off_path = os.path.join(dirpath, name)
                obj_name = name[:-4] + ".obj"
                obj_path = os.path.join(output_dir, obj_name)

                try:
                    mesh = trimesh.load(off_path, force='mesh')
                    mesh.export(obj_path)
                    print("Converted:", off_path, "->", obj_path)
                except Exception as e:
                    print("Failed:", off_path, "Error:", e)


def convert_modelnet40(root_dir):
    """
    root_dir: ModelNet40 根目录
    """
    for category in os.listdir(root_dir):
        category_dir = os.path.join(root_dir, category)
        if not os.path.isdir(category_dir):
            continue

        print(f"\nProcessing category: {category}")
        convert_split(category_dir, "train")
        convert_split(category_dir, "test")


if __name__ == "__main__":
    root_dir = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40"
    convert_modelnet40(root_dir)
    print("\nAll conversions done.")
