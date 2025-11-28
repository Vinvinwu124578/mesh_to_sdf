# import trimesh

# m = trimesh.load("model.off")
# m.export("model.obj")

import os
import trimesh

def batch_convert(input_dir, output_dir):
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    for dirpath, _, filenames in os.walk(input_dir):
        for name in filenames:
            if name.lower().endswith(".off"):
                off_path = os.path.join(dirpath, name)
                
                # 输出文件的路径 —— 全都放进 output_dir
                obj_name = name[:-4] + ".obj"
                obj_path = os.path.join(output_dir, obj_name)

                try:
                    mesh = trimesh.load(off_path)
                    mesh.export(obj_path)
                    print("Converted:", off_path, "->", obj_path)
                except Exception as e:
                    print("Failed:", off_path, "Error:", e)

if __name__ == "__main__":
    input_dir  = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/airplane/train"
    output_dir = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/airplane/train_obj"

    batch_convert(input_dir, output_dir)
    print("Done.")
