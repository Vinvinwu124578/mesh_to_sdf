from mesh_to_sdf import mesh_to_voxels

import trimesh
import skimage

mesh = trimesh.load("C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/airplane/train_obj/airplane_0001.obj")

voxels = mesh_to_voxels(mesh, 64, pad=True)

vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
mesh.show()