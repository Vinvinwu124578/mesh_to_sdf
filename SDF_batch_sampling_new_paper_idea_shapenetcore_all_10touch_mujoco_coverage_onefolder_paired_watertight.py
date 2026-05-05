import argparse
import concurrent.futures
import hashlib
import importlib.util
import multiprocessing
import os
import shutil
import subprocess
import sys
import traceback
import uuid
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from SDF_batch_sampling_new_paper_idea import (
    build_raycast_scene,
    compute_query_sdf_with_raycasting as compute_query_sdf_with_raycasting_legacy,
    sample_query_points_near_surface as sample_query_points_near_surface_legacy,
    sample_surface_points_for_storage,
)
from SDF_batch_sampling_new_paper_idea_shapenetcore_all import (
    find_shapenet_obj_files,
    iter_category_dirs,
)


TACTISTRUCT_PIPELINE_PATH = Path(
    r"C:\Users\wudaw\Downloads\Tactistruct-main\Tactistruct-main\pipeline_tactile_visualisation.py"
)
MANIFOLDPLUS_TMP_ROOT = Path(
    os.environ.get(
        "MANIFOLDPLUS_TMP_ROOT",
        str(Path(__file__).resolve().parent / "_manifoldplus_tmp"),
    )
)
_PIPELINE_MODULE = None


def load_tactistruct_pipeline_module():
    global _PIPELINE_MODULE
    if _PIPELINE_MODULE is not None:
        return _PIPELINE_MODULE

    if not TACTISTRUCT_PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Tactistruct pipeline file not found: {TACTISTRUCT_PIPELINE_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "tactistruct_pipeline_tactile_visualisation",
        str(TACTISTRUCT_PIPELINE_PATH),
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not create import spec for: {TACTISTRUCT_PIPELINE_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _PIPELINE_MODULE = module
    return module


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run coverage-aware MuJoCo tactile preprocessing for ShapeNetCore and save "
            "flat onefolder NPZ outputs. Uses process-based parallelism for stability."
        )
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=r"C:/Users/wudaw/Downloads/ShapeNetCore/ShapeNetCore",
    )
    parser.add_argument("--category-names", type=str, default=None)
    parser.add_argument("--max-objects-per-category", type=int, default=275)
    parser.add_argument(
        "--output-folder-name",
        type=str,
        default="tactistruct_npz_shapenet_mujoco_coverage_onefolder_paired_watertight_strict",
    )
    parser.add_argument(
        "--asset-folder-name",
        type=str,
        default="tactistruct_npz_shapenet_mujoco_coverage_assets_paired_watertight_strict",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 1))),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--num-tactile-samples", type=int, default=10)
    parser.add_argument("--tactile-num-fingers", type=int, default=10)
    parser.add_argument("--tactile-points-per-finger", type=int, default=3000)
    parser.add_argument("--dense-surface-sample-n", type=int, default=120000)
    parser.add_argument("--candidate-touch-samples", type=int, default=6000)
    parser.add_argument("--tactile-patch-radius-ratio", type=float, default=0.10)
    parser.add_argument("--tactile-min-touch-separation-ratio", type=float, default=0.055)
    parser.add_argument("--tactile-patch-thickness-ratio", type=float, default=0.035)
    parser.add_argument("--patch-min-normal-cos", type=float, default=0.05)
    parser.add_argument("--tactile-patch-dominant-normal-gap-cos", type=float, default=0.18)
    parser.add_argument("--tactile-patch-plane-gap-ratio", type=float, default=0.35)
    parser.add_argument("--tactile-patch-link-radius-ratio", type=float, default=0.0)
    parser.add_argument("--max-target-contact-offset-ratio", type=float, default=0.60)
    parser.add_argument("--tactile-reachable-clearance-ratio", type=float, default=0.92)
    parser.add_argument("--tactile-reachable-approach-steps", type=int, default=5)
    parser.add_argument("--normalization-bound", type=float, default=0.9)
    parser.add_argument("--num-surface-points", type=int, default=235000)
    parser.add_argument("--num-query-points", type=int, default=250000)
    parser.add_argument(
        "--query-uniform-region",
        type=str,
        default="cube",
        choices=("sphere", "cube"),
        help="Global random query support region mixed with near-surface samples. "
        "Use 'cube' to match inference over [-1, 1]^3.",
    )
    parser.add_argument(
        "--query-sampling-mode",
        type=str,
        default="paired_normal_offsets",
        choices=("paired_normal_offsets", "legacy_gaussian"),
        help="Use paired inside/outside normal offsets for balanced SDF signs, "
        "or the old isotropic Gaussian near-surface sampler.",
    )
    parser.add_argument(
        "--paired-query-fraction",
        type=float,
        default=0.90,
        help="Fraction of query points used for strict outside/inside pairs. "
        "The remaining points are uniform global samples.",
    )
    parser.add_argument("--paired-query-eps-min", type=float, default=0.002)
    parser.add_argument("--paired-query-eps-max", type=float, default=0.025)
    parser.add_argument("--paired-query-max-attempts", type=int, default=8)
    parser.add_argument(
        "--paired-query-anchor-mode",
        type=str,
        default="coverage_grid",
        choices=("uniform", "coverage_grid"),
        help="How to choose paired-query surface anchors. coverage_grid gives "
        "small spatial regions a minimum chance to produce red/blue pairs.",
    )
    parser.add_argument("--paired-query-coverage-grid-size", type=int, default=12)
    parser.add_argument("--paired-query-coverage-min-per-cell", type=int, default=1)
    parser.add_argument(
        "--paired-query-eps-retries",
        type=int,
        default=3,
        help="Number of offset distances to try for each anchor before rejecting "
        "it against the watertight sign proxy.",
    )
    parser.add_argument(
        "--watertight-proxy-mode",
        type=str,
        default="repair",
        choices=("repair", "poisson", "pymeshlab_poisson", "pymeshfix", "manifoldplus", "convex_hull", "none"),
        help="Mesh used only for inside/outside signs. 'repair' is fast; "
        "'pymeshlab_poisson' is a higher-quality MeshLab Screened Poisson path.",
    )
    parser.add_argument(
        "--watertight-mesh-usage",
        type=str,
        default="sign_proxy",
        choices=("sign_proxy", "full_pipeline"),
        help="Use the watertight mesh only for query inside/outside signs, "
        "or replace the normalized mesh before STL export, surface sampling, "
        "query sampling, and MuJoCo touch simulation.",
    )
    parser.add_argument(
        "--non-watertight-policy",
        type=str,
        default="skip",
        choices=("paired_normal_fallback", "skip"),
        help="What to do when no watertight sign proxy can be built.",
    )
    parser.add_argument("--proxy-poisson-samples", type=int, default=50000)
    parser.add_argument("--proxy-poisson-depth", type=int, default=8)
    parser.add_argument("--proxy-poisson-full-depth", type=int, default=5)
    parser.add_argument("--proxy-poisson-threads", type=int, default=8)
    parser.add_argument("--manifoldplus-path", type=str, default=None)
    parser.add_argument(
        "--manifoldplus-depth",
        type=int,
        default=8,
        help="Octree depth passed to ManifoldPlus. Higher values preserve more "
        "detail but increase runtime and output face count.",
    )
    parser.add_argument(
        "--mujoco-max-faces",
        type=int,
        default=190000,
        help="Maximum face count for the STL exported to MuJoCo. MuJoCo's STL "
        "loader rejects meshes above 200000 faces.",
    )
    parser.add_argument("--query-occupancy-nsamples", type=int, default=11)
    parser.add_argument("--query-near-surface-sign-band", type=float, default=0.01)
    parser.add_argument("--base-seed", type=int, default=42)
    return parser.parse_args()


def parse_category_names(value):
    if value is None:
        return None
    names = [item.strip() for item in str(value).split(",")]
    names = [item for item in names if item]
    return names or None


def object_seed(base_seed, obj_path):
    digest = hashlib.sha1(str(obj_path).encode("utf-8")).hexdigest()[:8]
    return int(base_seed) + int(digest, 16)


def windows_path_to_wsl(path):
    windows_path = Path(path).resolve().as_posix()
    result = subprocess.run(
        ["wsl", "-d", "Ubuntu", "wslpath", "-a", windows_path],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to convert Windows path to WSL path: "
            f"{windows_path}. {result.stderr or result.stdout}"
        )
    return result.stdout.strip()


def load_simple_obj_mesh(path):
    vertices = []
    faces = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                indices = []
                for token in line.split()[1:]:
                    vertex_token = token.split("/")[0]
                    if not vertex_token:
                        continue
                    index = int(vertex_token)
                    if index < 0:
                        index = len(vertices) + index
                    else:
                        index = index - 1
                    indices.append(index)
                if len(indices) == 3:
                    faces.append(indices)
                elif len(indices) > 3:
                    root = indices[0]
                    for offset in range(1, len(indices) - 1):
                        faces.append([root, indices[offset], indices[offset + 1]])

    if not vertices or not faces:
        raise RuntimeError(f"OBJ mesh is empty or has no faces: {path}")
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def build_flat_output_path(obj_path, root_dir, output_folder_name):
    rel_path = os.path.relpath(obj_path, root_dir)
    rel_without_ext = os.path.splitext(rel_path)[0]
    safe_name = rel_without_ext.replace("\\", "__").replace("/", "__").replace(":", "_")
    out_dir = os.path.join(root_dir, output_folder_name)
    return os.path.join(out_dir, safe_name + ".npz")


def build_asset_export_path(obj_path, root_dir, asset_folder_name):
    rel_path = os.path.relpath(obj_path, root_dir)
    rel_without_ext = os.path.splitext(rel_path)[0]
    safe_name = rel_without_ext.replace("\\", "__").replace("/", "__").replace(":", "_")
    asset_dir = os.path.join(root_dir, asset_folder_name)
    return os.path.join(asset_dir, safe_name + "__normalized.stl")


def compute_bbox_diag(points):
    points = np.asarray(points, dtype=np.float32)
    mn = np.min(points, axis=0)
    mx = np.max(points, axis=0)
    return float(np.linalg.norm(mx - mn))


def sample_dense_surface_points(mesh, sample_count):
    points, face_ids = trimesh.sample.sample_surface(mesh, int(sample_count))
    points = points.astype(np.float32)
    normals = mesh.face_normals[face_ids].astype(np.float32)

    centroid = mesh.bounding_box.centroid.astype(np.float32)
    outward_hint = points - centroid[None, :]
    flip_mask = np.einsum("ij,ij->i", normals, outward_hint) < 0.0
    normals[flip_mask] *= -1.0
    return points, normals


def compute_scene_distance(scene, points, batch_size=65536):
    import open3d as o3d

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    distances = np.zeros((len(points),), dtype=np.float32)

    for start in range(0, len(points), int(batch_size)):
        end = min(len(points), start + int(batch_size))
        tensor = o3d.core.Tensor(
            points[start:end],
            dtype=o3d.core.Dtype.Float32,
        )
        distances[start:end] = scene.compute_distance(tensor).numpy().astype(np.float32)

    return distances


def compute_scene_occupancy(scene, points, nsamples=11, batch_size=65536):
    import open3d as o3d

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    occupancy = np.zeros((len(points),), dtype=np.float32)
    nsamples = max(1, int(nsamples))
    if nsamples % 2 == 0:
        nsamples += 1

    for start in range(0, len(points), int(batch_size)):
        end = min(len(points), start + int(batch_size))
        tensor = o3d.core.Tensor(
            points[start:end],
            dtype=o3d.core.Dtype.Float32,
        )
        occupancy[start:end] = scene.compute_occupancy(
            tensor,
            nsamples=nsamples,
        ).numpy().astype(np.float32)

    return occupancy


def normalize_rows(vectors, eps=1e-8):
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, float(eps), None)


def orient_normals_outward_from_center(points, normals, center):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    normals = normalize_rows(normals)
    center = np.asarray(center, dtype=np.float32).reshape(1, 3)
    outward_hint = points - center
    flip_mask = np.einsum("ij,ij->i", normals, outward_hint) < 0.0
    oriented = normals.copy()
    oriented[flip_mask] *= -1.0
    return oriented.astype(np.float32)


def sample_uniform_query_points(count, uniform_region="cube", rng=None):
    count = int(count)
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    if rng is None:
        rng = np.random.default_rng()

    resolved_region = str(uniform_region).strip().lower()
    if resolved_region == "cube":
        return rng.uniform(-1.0, 1.0, size=(count, 3)).astype(np.float32)
    if resolved_region != "sphere":
        raise ValueError(
            f"Unsupported uniform_region={uniform_region!r}. Expected 'sphere' or 'cube'."
        )

    points = []
    remaining = count
    while remaining > 0:
        candidate_count = max(remaining * 2, 1024)
        candidates = rng.uniform(-1.0, 1.0, size=(candidate_count, 3)).astype(np.float32)
        keep = np.sum(candidates * candidates, axis=1) <= 1.0
        kept = candidates[keep][:remaining]
        if len(kept) > 0:
            points.append(kept)
            remaining -= len(kept)
    return np.concatenate(points, axis=0).astype(np.float32)


def build_surface_grid_strata(surface_points, grid_size=12):
    surface_points = np.asarray(surface_points, dtype=np.float32).reshape(-1, 3)
    grid_size = max(1, int(grid_size))
    if len(surface_points) == 0 or grid_size <= 1:
        return None

    bounds_min = np.min(surface_points, axis=0)
    bounds_max = np.max(surface_points, axis=0)
    extent = np.maximum(bounds_max - bounds_min, 1e-6)
    scaled = (surface_points - bounds_min[None, :]) / extent[None, :]
    coords = np.floor(scaled * float(grid_size)).astype(np.int64)
    coords = np.clip(coords, 0, grid_size - 1)
    keys = (
        coords[:, 0] * int(grid_size) * int(grid_size)
        + coords[:, 1] * int(grid_size)
        + coords[:, 2]
    )

    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable").astype(np.int64)
    starts = np.zeros((len(counts),), dtype=np.int64)
    if len(counts) > 1:
        starts[1:] = np.cumsum(counts[:-1], dtype=np.int64)
    return {
        "order": order,
        "starts": starts,
        "counts": counts.astype(np.int64),
        "stratum_count": int(len(counts)),
    }


def sample_surface_anchor_ids(
    surface_points,
    count,
    rng,
    anchor_mode="uniform",
    strata=None,
    min_per_cell=1,
):
    surface_points = np.asarray(surface_points, dtype=np.float32).reshape(-1, 3)
    count = int(count)
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if len(surface_points) == 0:
        raise ValueError("surface_points is empty; cannot sample paired-query anchors.")

    anchor_mode = str(anchor_mode).strip().lower()
    if anchor_mode != "coverage_grid" or strata is None:
        return rng.choice(
            len(surface_points),
            size=count,
            replace=len(surface_points) < count,
        ).astype(np.int64)

    counts = np.asarray(strata["counts"], dtype=np.int64)
    starts = np.asarray(strata["starts"], dtype=np.int64)
    order = np.asarray(strata["order"], dtype=np.int64)
    active_strata = np.flatnonzero(counts > 0)
    if len(active_strata) == 0:
        return rng.choice(
            len(surface_points),
            size=count,
            replace=len(surface_points) < count,
        ).astype(np.int64)

    selected_chunks = []
    guaranteed = min(count, len(active_strata) * max(0, int(min_per_cell)))
    if guaranteed > 0:
        repeats = int(np.ceil(float(guaranteed) / float(len(active_strata))))
        selected = []
        for _ in range(repeats):
            selected.append(rng.permutation(active_strata))
        selected_chunks.append(np.concatenate(selected, axis=0)[:guaranteed])

    remaining = count - guaranteed
    if remaining > 0:
        # sqrt(area-count) keeps large regions likely while giving small parts a voice.
        weights = np.sqrt(counts[active_strata].astype(np.float64))
        weights = weights / np.sum(weights)
        selected_chunks.append(
            rng.choice(active_strata, size=remaining, replace=True, p=weights)
        )

    selected_strata = np.concatenate(selected_chunks, axis=0).astype(np.int64)
    offsets = (
        rng.random(len(selected_strata)) * counts[selected_strata].astype(np.float64)
    ).astype(np.int64)
    return order[starts[selected_strata] + offsets].astype(np.int64)


def clean_mesh_for_sign_proxy(mesh):
    proxy = mesh.copy()
    for method_name in (
        "remove_degenerate_faces",
        "remove_duplicate_faces",
        "remove_infinite_values",
        "remove_unreferenced_vertices",
    ):
        method = getattr(proxy, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass

    try:
        trimesh.repair.fix_winding(proxy)
    except Exception:
        pass
    try:
        trimesh.repair.fill_holes(proxy)
    except Exception:
        pass
    try:
        trimesh.repair.fix_normals(proxy, multibody=True)
    except Exception:
        pass
    try:
        trimesh.repair.fix_inversion(proxy, multibody=True)
    except Exception:
        pass
    try:
        proxy.remove_unreferenced_vertices()
    except Exception:
        pass
    return proxy


def simplify_mesh_for_mujoco(mesh, max_faces=190000):
    max_faces = int(max_faces)
    if max_faces <= 0 or len(mesh.faces) <= max_faces:
        return mesh.copy(), False

    try:
        import pymeshlab

        meshset = pymeshlab.MeshSet()
        meshset.add_mesh(
            pymeshlab.Mesh(
                vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
                face_matrix=np.asarray(mesh.faces, dtype=np.int32),
            ),
            "mujoco_export_mesh",
        )
        meshset.meshing_decimation_quadric_edge_collapse(
            targetfacenum=max_faces,
            preservenormal=True,
            qualitythr=0.3,
        )
        current = meshset.current_mesh()
        simplified = trimesh.Trimesh(
            vertices=np.asarray(current.vertex_matrix(), dtype=np.float32),
            faces=np.asarray(current.face_matrix(), dtype=np.int64),
            process=True,
        )
        simplified = clean_mesh_for_sign_proxy(simplified)
        if len(simplified.faces) > 0 and len(simplified.faces) <= max_faces:
            return simplified, True
        print(
            f"[WARN] PyMeshLab decimation returned {len(simplified.faces)} faces; "
            "trying trimesh decimation."
        )
    except Exception as exc:
        print(f"[WARN] PyMeshLab decimation for MuJoCo export failed: {exc}")

    try:
        simplified = mesh.simplify_quadric_decimation(
            face_count=max_faces,
            aggression=5,
        )
        simplified = clean_mesh_for_sign_proxy(simplified)
        if len(simplified.faces) > 0 and len(simplified.faces) <= max_faces:
            return simplified, True
        print(
            f"[WARN] quadric decimation returned {len(simplified.faces)} faces; "
            "falling back to random face subset for MuJoCo export."
        )
    except Exception as exc:
        print(f"[WARN] quadric decimation for MuJoCo export failed: {exc}")

    face_ids = np.linspace(0, len(mesh.faces) - 1, num=max_faces, dtype=np.int64)
    subset = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces)[face_ids],
        process=True,
    )
    return clean_mesh_for_sign_proxy(subset), True


def build_poisson_sign_proxy(surface_points, surface_normals, sample_count, depth, seed):
    import open3d as o3d

    points = np.asarray(surface_points, dtype=np.float32).reshape(-1, 3)
    normals = normalize_rows(surface_normals)
    if len(points) == 0:
        raise RuntimeError("Cannot build a Poisson proxy from an empty surface.")

    rng = np.random.default_rng(int(seed))
    sample_count = min(len(points), max(1000, int(sample_count)))
    chosen_ids = rng.choice(len(points), size=sample_count, replace=False)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[chosen_ids].astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(normals[chosen_ids].astype(np.float64))

    proxy_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=max(5, int(depth)),
    )
    # Cropping a Poisson surface can open the shell and break watertightness.
    # Keep the closed proxy intact because it is used only for sign queries.
    proxy_o3d.compute_triangle_normals()

    vertices = np.asarray(proxy_o3d.vertices, dtype=np.float32)
    faces = np.asarray(proxy_o3d.triangles, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError("Poisson proxy reconstruction produced an empty mesh.")

    proxy = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    return clean_mesh_for_sign_proxy(proxy)


def build_pymeshlab_poisson_sign_proxy(
    surface_points,
    surface_normals,
    sample_count,
    depth,
    full_depth,
    threads,
    seed,
):
    import pymeshlab

    points = np.asarray(surface_points, dtype=np.float32).reshape(-1, 3)
    normals = normalize_rows(surface_normals)
    if len(points) == 0:
        raise RuntimeError("Cannot build a PyMeshLab proxy from an empty surface.")

    rng = np.random.default_rng(int(seed))
    sample_count = min(len(points), max(1000, int(sample_count)))
    chosen_ids = rng.choice(len(points), size=sample_count, replace=False)

    point_cloud = pymeshlab.Mesh(
        vertex_matrix=points[chosen_ids].astype(np.float64),
        v_normals_matrix=normals[chosen_ids].astype(np.float64),
    )
    meshset = pymeshlab.MeshSet()
    meshset.add_mesh(point_cloud, "oriented_surface_points")
    meshset.generate_surface_reconstruction_screened_poisson(
        depth=max(5, int(depth)),
        fulldepth=max(0, int(full_depth)),
        preclean=True,
        threads=max(1, int(threads)),
    )

    reconstructed = meshset.current_mesh()
    vertices = np.asarray(reconstructed.vertex_matrix(), dtype=np.float32)
    faces = np.asarray(reconstructed.face_matrix(), dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError("PyMeshLab Poisson reconstruction produced an empty mesh.")

    proxy = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    return clean_mesh_for_sign_proxy(proxy)


def build_manifoldplus_sign_proxy(mesh, manifoldplus_path, seed, depth=8):
    if manifoldplus_path is None or not str(manifoldplus_path).strip():
        executable = shutil.which("manifold") or shutil.which("manifoldplus") or shutil.which("ManifoldPlus")
    else:
        executable = str(manifoldplus_path)

    use_wsl = executable is not None and str(executable).startswith("wsl:")
    if executable is None or (not use_wsl and not os.path.exists(executable)):
        raise FileNotFoundError(
            "ManifoldPlus executable was not found. Pass --manifoldplus-path "
            "or put the executable on PATH. For a WSL build, pass "
            "--manifoldplus-path wsl:/absolute/linux/path/to/manifold."
        )

    MANIFOLDPLUS_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_dir = MANIFOLDPLUS_TMP_ROOT / f"manifoldplus_{int(seed)}_{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=False, exist_ok=False)
    try:
        input_path = tmp_dir / "input.obj"
        output_path = tmp_dir / "output.obj"
        mesh.export(input_path)

        if use_wsl:
            linux_executable = str(executable)[len("wsl:") :]
            command = [
                "wsl",
                "-d",
                "Ubuntu",
                linux_executable,
                "--input",
                windows_path_to_wsl(input_path),
                "--output",
                windows_path_to_wsl(output_path),
                "--depth",
                str(max(1, int(depth))),
            ]
        else:
            command = [
                executable,
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--depth",
                str(max(1, int(depth))),
            ]
        result = subprocess.run(
            command,
            cwd=str(tmp_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "ManifoldPlus failed with exit code "
                f"{result.returncode}: {result.stderr or result.stdout}"
            )
        if not output_path.exists():
            raise RuntimeError("ManifoldPlus completed but did not create output.obj.")
        proxy = load_simple_obj_mesh(output_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return proxy


def build_pymeshfix_sign_proxy(mesh):
    import pymeshfix

    fixer = pymeshfix.MeshFix(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int32),
    )
    fixer.repair(
        verbose=False,
        joincomp=True,
        remove_smallest_components=False,
    )
    proxy = trimesh.Trimesh(
        vertices=np.asarray(fixer.points, dtype=np.float32),
        faces=np.asarray(fixer.faces, dtype=np.int64),
        process=True,
    )
    return clean_mesh_for_sign_proxy(proxy)


def build_sign_proxy_mesh(
    mesh,
    surface_points,
    surface_normals,
    mode="repair",
    poisson_sample_count=50000,
    poisson_depth=8,
    poisson_full_depth=5,
    poisson_threads=8,
    manifoldplus_path=None,
    manifoldplus_depth=8,
    seed=42,
):
    mode = str(mode).strip().lower()
    if mesh.is_watertight:
        return mesh, "original_watertight"
    if mode == "none":
        return None, "none"

    if mode in ("repair", "poisson", "pymeshlab_poisson", "pymeshfix", "manifoldplus"):
        repaired = clean_mesh_for_sign_proxy(mesh)
        if repaired.is_watertight:
            return repaired, "trimesh_repair"

    if mode == "poisson":
        try:
            poisson_proxy = build_poisson_sign_proxy(
                surface_points=surface_points,
                surface_normals=surface_normals,
                sample_count=poisson_sample_count,
                depth=poisson_depth,
                seed=seed,
            )
            if poisson_proxy.is_watertight:
                return poisson_proxy, "poisson"
            return poisson_proxy, "poisson_non_watertight"
        except Exception as exc:
            print(f"[WARN] Poisson watertight proxy failed: {exc}")

    if mode == "pymeshlab_poisson":
        try:
            pymeshlab_proxy = build_pymeshlab_poisson_sign_proxy(
                surface_points=surface_points,
                surface_normals=surface_normals,
                sample_count=poisson_sample_count,
                depth=poisson_depth,
                full_depth=poisson_full_depth,
                threads=poisson_threads,
                seed=seed,
            )
            if pymeshlab_proxy.is_watertight:
                return pymeshlab_proxy, "pymeshlab_poisson"
            return pymeshlab_proxy, "pymeshlab_poisson_non_watertight"
        except Exception as exc:
            print(f"[WARN] PyMeshLab Poisson watertight proxy failed: {exc}")

    if mode == "manifoldplus":
        try:
            manifold_proxy = build_manifoldplus_sign_proxy(
                mesh=mesh,
                manifoldplus_path=manifoldplus_path,
                depth=manifoldplus_depth,
                seed=seed,
            )
            if manifold_proxy.is_watertight:
                return manifold_proxy, "manifoldplus"
            return manifold_proxy, "manifoldplus_non_watertight"
        except Exception as exc:
            print(f"[WARN] ManifoldPlus watertight proxy failed: {exc}")

    if mode == "pymeshfix":
        try:
            meshfix_proxy = build_pymeshfix_sign_proxy(mesh)
            if meshfix_proxy.is_watertight:
                return meshfix_proxy, "pymeshfix"
            return meshfix_proxy, "pymeshfix_non_watertight"
        except Exception as exc:
            print(f"[WARN] PyMeshFix watertight proxy failed: {exc}")

    if mode == "convex_hull":
        try:
            hull = clean_mesh_for_sign_proxy(mesh.convex_hull)
            if hull.is_watertight:
                return hull, "convex_hull"
        except Exception as exc:
            print(f"[WARN] Convex hull watertight proxy failed: {exc}")

    return None, "none"


def sample_paired_normal_query_points(
    surface_points,
    surface_normals,
    number_of_points,
    uniform_region,
    paired_fraction,
    eps_min,
    eps_max,
    rng,
    sign_scene=None,
    sign_proxy_is_watertight=False,
    occupancy_nsamples=11,
    max_attempts=8,
    anchor_mode="coverage_grid",
    coverage_grid_size=12,
    coverage_min_per_cell=1,
    eps_retries=3,
):
    surface_points = np.asarray(surface_points, dtype=np.float32).reshape(-1, 3)
    surface_normals = normalize_rows(surface_normals)
    number_of_points = int(number_of_points)
    paired_fraction = float(np.clip(float(paired_fraction), 0.0, 1.0))
    paired_point_count = int(number_of_points * paired_fraction)
    paired_point_count -= paired_point_count % 2
    target_pairs = paired_point_count // 2

    eps_min = max(float(eps_min), 1e-6)
    eps_max = max(float(eps_max), eps_min)
    validate_with_proxy = sign_scene is not None and bool(sign_proxy_is_watertight)
    anchor_mode = str(anchor_mode).strip().lower()
    strata = None
    if anchor_mode == "coverage_grid":
        strata = build_surface_grid_strata(
            surface_points,
            grid_size=coverage_grid_size,
        )
    stratum_count = 0 if strata is None else int(strata["stratum_count"])
    eps_retries = max(1, int(eps_retries))

    outside_chunks = []
    inside_chunks = []
    anchor_chunks = []
    eps_chunks = []
    accepted_pairs = 0
    rejected_pairs = 0
    flipped_pairs = 0
    proxy_tested_pairs = 0
    fallback_pairs = 0

    attempts = 0
    while accepted_pairs < target_pairs and attempts < max(1, int(max_attempts)):
        attempts += 1
        remaining = target_pairs - accepted_pairs
        sample_count = remaining
        if validate_with_proxy:
            sample_count = max(remaining * 2, min(len(surface_points), remaining + 2048))

        anchor_ids = sample_surface_anchor_ids(
            surface_points=surface_points,
            count=sample_count,
            rng=rng,
            anchor_mode=anchor_mode,
            strata=strata,
            min_per_cell=coverage_min_per_cell,
        )
        anchors = surface_points[anchor_ids]
        normals = surface_normals[anchor_ids]

        if validate_with_proxy:
            accepted_outside_chunks = []
            accepted_inside_chunks = []
            accepted_anchor_chunks = []
            accepted_eps_chunks = []

            pending_anchors = anchors
            pending_normals = normals
            pending_anchor_ids = anchor_ids
            for _ in range(eps_retries):
                pending_count = len(pending_anchor_ids)
                if pending_count <= 0:
                    break

                eps = rng.uniform(
                    eps_min,
                    eps_max,
                    size=(pending_count, 1),
                ).astype(np.float32)
                outside = pending_anchors + eps * pending_normals
                inside = pending_anchors - eps * pending_normals
                occupancy = compute_scene_occupancy(
                    sign_scene,
                    np.concatenate([outside, inside], axis=0),
                    nsamples=occupancy_nsamples,
                )
                proxy_tested_pairs += int(pending_count)
                outside_inside = occupancy[:pending_count] > 0.5
                inside_inside = occupancy[pending_count:] > 0.5

                valid = (~outside_inside) & inside_inside
                flipped = outside_inside & (~inside_inside)
                accepted_mask = valid | flipped
                flipped_pairs += int(np.count_nonzero(flipped))

                if np.any(valid):
                    accepted_outside_chunks.append(outside[valid])
                    accepted_inside_chunks.append(inside[valid])
                    accepted_anchor_chunks.append(pending_anchor_ids[valid])
                    accepted_eps_chunks.append(eps[valid].reshape(-1))
                if np.any(flipped):
                    accepted_outside_chunks.append(inside[flipped])
                    accepted_inside_chunks.append(outside[flipped])
                    accepted_anchor_chunks.append(pending_anchor_ids[flipped])
                    accepted_eps_chunks.append(eps[flipped].reshape(-1))

                retry_mask = ~accepted_mask
                pending_anchors = pending_anchors[retry_mask]
                pending_normals = pending_normals[retry_mask]
                pending_anchor_ids = pending_anchor_ids[retry_mask]

            rejected_pairs += int(len(pending_anchor_ids))
            if accepted_outside_chunks:
                accepted_outside = np.concatenate(accepted_outside_chunks, axis=0)
                accepted_inside = np.concatenate(accepted_inside_chunks, axis=0)
                accepted_anchor_ids = np.concatenate(accepted_anchor_chunks, axis=0)
                accepted_eps = np.concatenate(accepted_eps_chunks, axis=0)
            else:
                accepted_outside = np.zeros((0, 3), dtype=np.float32)
                accepted_inside = np.zeros((0, 3), dtype=np.float32)
                accepted_anchor_ids = np.zeros((0,), dtype=np.int64)
                accepted_eps = np.zeros((0,), dtype=np.float32)
        else:
            eps = rng.uniform(eps_min, eps_max, size=(sample_count, 1)).astype(np.float32)
            outside = anchors + eps * normals
            inside = anchors - eps * normals
            accepted_outside = outside
            accepted_inside = inside
            accepted_anchor_ids = anchor_ids
            accepted_eps = eps.reshape(-1)

        keep_count = min(remaining, len(accepted_outside))
        if keep_count <= 0:
            continue

        outside_chunks.append(accepted_outside[:keep_count].astype(np.float32))
        inside_chunks.append(accepted_inside[:keep_count].astype(np.float32))
        anchor_chunks.append(accepted_anchor_ids[:keep_count].astype(np.int32))
        eps_chunks.append(accepted_eps[:keep_count].astype(np.float32))
        accepted_pairs += int(keep_count)

    if accepted_pairs < target_pairs:
        remaining = target_pairs - accepted_pairs
        print(
            f"[WARN] paired proxy validation accepted only {accepted_pairs}/{target_pairs} "
            f"pairs; topping up {remaining} pairs with normal-offset labels."
        )
        anchor_ids = sample_surface_anchor_ids(
            surface_points=surface_points,
            count=remaining,
            rng=rng,
            anchor_mode=anchor_mode,
            strata=strata,
            min_per_cell=coverage_min_per_cell,
        )
        anchors = surface_points[anchor_ids]
        normals = surface_normals[anchor_ids]
        eps = rng.uniform(eps_min, eps_max, size=(remaining, 1)).astype(np.float32)
        outside_chunks.append((anchors + eps * normals).astype(np.float32))
        inside_chunks.append((anchors - eps * normals).astype(np.float32))
        anchor_chunks.append(anchor_ids.astype(np.int32))
        eps_chunks.append(eps.reshape(-1).astype(np.float32))
        accepted_pairs += int(remaining)
        fallback_pairs += int(remaining)

    if accepted_pairs > 0:
        outside_points = np.concatenate(outside_chunks, axis=0)
        inside_points = np.concatenate(inside_chunks, axis=0)
        anchor_ids = np.concatenate(anchor_chunks, axis=0)
        pair_eps = np.concatenate(eps_chunks, axis=0)
        paired_points = np.empty((accepted_pairs * 2, 3), dtype=np.float32)
        paired_points[0::2] = outside_points
        paired_points[1::2] = inside_points
        pair_ids = np.repeat(np.arange(accepted_pairs, dtype=np.int32), 2)
        pair_side = np.tile(np.asarray([1, -1], dtype=np.int8), accepted_pairs)
        pair_anchor_ids = np.repeat(anchor_ids.astype(np.int32), 2)
        pair_eps_values = np.repeat(pair_eps.astype(np.float32), 2)
    else:
        paired_points = np.zeros((0, 3), dtype=np.float32)
        pair_ids = np.zeros((0,), dtype=np.int32)
        pair_side = np.zeros((0,), dtype=np.int8)
        pair_anchor_ids = np.zeros((0,), dtype=np.int32)
        pair_eps_values = np.zeros((0,), dtype=np.float32)

    uniform_count = max(0, number_of_points - len(paired_points))
    uniform_points = sample_uniform_query_points(
        uniform_count,
        uniform_region=uniform_region,
        rng=rng,
    )
    query_points = np.concatenate([paired_points, uniform_points], axis=0).astype(np.float32)
    query_pair_ids = np.concatenate(
        [pair_ids, np.full((uniform_count,), -1, dtype=np.int32)],
        axis=0,
    )
    query_pair_side = np.concatenate(
        [pair_side, np.zeros((uniform_count,), dtype=np.int8)],
        axis=0,
    )
    query_anchor_ids = np.concatenate(
        [pair_anchor_ids, np.full((uniform_count,), -1, dtype=np.int32)],
        axis=0,
    )
    query_pair_eps = np.concatenate(
        [pair_eps_values, np.zeros((uniform_count,), dtype=np.float32)],
        axis=0,
    )

    metadata = {
        "requested_pairs": int(target_pairs),
        "accepted_pairs": int(accepted_pairs),
        "rejected_pairs": int(rejected_pairs),
        "flipped_pairs": int(flipped_pairs),
        "proxy_tested_pairs": int(proxy_tested_pairs),
        "fallback_pairs": int(fallback_pairs),
        "uniform_count": int(uniform_count),
        "validated_with_proxy": bool(validate_with_proxy),
        "anchor_mode": str(anchor_mode),
        "anchor_strata_count": int(stratum_count),
        "eps_retries": int(eps_retries),
    }
    return query_points, query_pair_ids, query_pair_side, query_anchor_ids, query_pair_eps, metadata


def assign_query_sdf_signs(
    distance_scene,
    sign_scene,
    query_points,
    query_pair_side,
    sign_proxy_is_watertight,
    occupancy_nsamples,
    surface_points,
    surface_normals,
    near_surface_sign_band,
):
    unsigned_distance = compute_scene_distance(distance_scene, query_points)
    query_sdf = unsigned_distance.copy()
    query_pair_side = np.asarray(query_pair_side, dtype=np.int8).reshape(-1)

    if sign_scene is not None and bool(sign_proxy_is_watertight):
        occupancy = compute_scene_occupancy(
            sign_scene,
            query_points,
            nsamples=occupancy_nsamples,
        )
        inside_mask = occupancy > 0.5
        query_sdf[inside_mask] *= -1.0
        return query_sdf.astype(np.float32), {
            "sign_source": "watertight_proxy_occupancy",
            "inside_count": int(np.count_nonzero(inside_mask)),
            "outside_count": int(len(query_sdf) - np.count_nonzero(inside_mask)),
        }

    inside_pair_mask = query_pair_side < 0
    query_sdf[inside_pair_mask] *= -1.0

    unknown_mask = query_pair_side == 0
    if np.any(unknown_mask) and surface_points is not None and surface_normals is not None:
        nearest_tree = cKDTree(np.asarray(surface_points, dtype=np.float32).reshape(-1, 3))
        _, nearest_ids = nearest_tree.query(query_points[unknown_mask], k=1)
        normals = normalize_rows(surface_normals)[nearest_ids]
        direction_from_surface = query_points[unknown_mask] - surface_points[nearest_ids]
        local_inside_mask = np.einsum("ij,ij->i", direction_from_surface, normals) < 0
        near_surface_mask = unsigned_distance[unknown_mask] <= float(near_surface_sign_band)
        unknown_ids = np.flatnonzero(unknown_mask)
        query_sdf[unknown_ids[near_surface_mask & local_inside_mask]] *= -1.0

    return query_sdf.astype(np.float32), {
        "sign_source": "paired_normal_labels_with_local_uniform_fallback",
        "inside_count": int(np.count_nonzero(query_sdf < 0.0)),
        "outside_count": int(np.count_nonzero(query_sdf >= 0.0)),
    }


def compute_sphere_reachability_mask(
    scene,
    surface_points,
    surface_normals,
    probe_radius,
    approach_offset,
    clearance_ratio=0.92,
    approach_steps=5,
):
    surface_points = np.asarray(surface_points, dtype=np.float32).reshape(-1, 3)
    surface_normals = np.asarray(surface_normals, dtype=np.float32).reshape(-1, 3)

    if len(surface_points) == 0:
        return np.zeros((0,), dtype=bool)

    normal_norm = np.linalg.norm(surface_normals, axis=1, keepdims=True)
    surface_normals = surface_normals / np.clip(normal_norm, 1e-8, None)

    start_offset = float(probe_radius)
    end_offset = float(probe_radius) + max(float(approach_offset), 0.0)
    step_count = max(2, int(approach_steps))
    center_offsets = np.linspace(
        start_offset,
        end_offset,
        num=step_count,
        endpoint=True,
        dtype=np.float32,
    )

    clearance_threshold = float(clearance_ratio) * float(probe_radius)
    reachable_mask = np.ones((len(surface_points),), dtype=bool)

    for center_offset in center_offsets:
        probe_centers = surface_points + surface_normals * float(center_offset)
        clearance = compute_scene_distance(scene, probe_centers)
        reachable_mask &= clearance >= clearance_threshold
        if not np.any(reachable_mask):
            break

    return reachable_mask


def normalize_vector(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-8:
        return np.zeros((3,), dtype=np.float32)
    return (vector / norm).astype(np.float32)


def build_tangent_basis(normal):
    normal = normalize_vector(normal)
    if abs(float(normal[2])) < 0.9:
        helper = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        helper = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    tangent_u = np.cross(normal, helper).astype(np.float32)
    tangent_u = normalize_vector(tangent_u)
    tangent_v = np.cross(normal, tangent_u).astype(np.float32)
    tangent_v = normalize_vector(tangent_v)
    return tangent_u, tangent_v


def compute_patch_tangent_geometry(points, center_point, center_normal):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    center_point = np.asarray(center_point, dtype=np.float32).reshape(3)
    center_normal = normalize_vector(center_normal)
    tangent_u, tangent_v = build_tangent_basis(center_normal)
    offsets = points - center_point[None, :]
    uv = np.stack(
        [
            offsets @ tangent_u,
            offsets @ tangent_v,
        ],
        axis=1,
    ).astype(np.float32)

    if len(uv) <= 1:
        principal_coords = uv.astype(np.float32)
        major_extent = float(np.max(np.abs(principal_coords[:, 0]))) if len(principal_coords) else 0.0
        minor_extent = float(np.max(np.abs(principal_coords[:, 1]))) if len(principal_coords) else 0.0
        rotation = np.eye(2, dtype=np.float32)
    else:
        cov = np.cov(uv.T).astype(np.float32)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order].astype(np.float32)
        principal_coords = (uv @ eigvecs).astype(np.float32)
        major_extent = float(np.quantile(np.abs(principal_coords[:, 0]), 0.9))
        minor_extent = float(np.quantile(np.abs(principal_coords[:, 1]), 0.9))
        rotation = eigvecs

    return {
        "uv": uv.astype(np.float32),
        "principal_coords": principal_coords.astype(np.float32),
        "major_extent": float(max(major_extent, 0.0)),
        "minor_extent": float(max(minor_extent, 0.0)),
        "rotation": rotation.astype(np.float32),
        "tangent_u": tangent_u.astype(np.float32),
        "tangent_v": tangent_v.astype(np.float32),
    }


def align_normal_to_reference(normal, reference_normal):
    normal = normalize_vector(normal)
    reference_normal = normalize_vector(reference_normal)
    if np.dot(normal, reference_normal) < 0.0:
        normal = -normal
    return normal.astype(np.float32)


def estimate_patch_ids(dense_points, dense_tree, center_point, patch_radius, nearest_fallback_k=256):
    patch_ids = dense_tree.query_ball_point(center_point, r=float(patch_radius))
    if patch_ids:
        return np.asarray(patch_ids, dtype=np.int64)

    nearest_k = min(int(nearest_fallback_k), len(dense_points))
    _, nearest_ids = dense_tree.query(center_point[None, :], k=nearest_k)
    return np.asarray(nearest_ids, dtype=np.int64).reshape(-1)


def keep_patch_component_near_center(points, center_point, link_radius):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(points) <= 1 or link_radius is None or float(link_radius) <= 0.0:
        return np.arange(len(points), dtype=np.int64)

    local_tree = cKDTree(points)
    _, anchor_local = local_tree.query(center_point[None, :], k=1)
    anchor_local = int(np.asarray(anchor_local).reshape(-1)[0])

    visited = np.zeros((len(points),), dtype=bool)
    queue = [anchor_local]
    visited[anchor_local] = True

    while queue:
        current = queue.pop()
        neighbor_ids = local_tree.query_ball_point(points[current], r=float(link_radius))
        for neighbor_id in neighbor_ids:
            neighbor_id = int(neighbor_id)
            if not visited[neighbor_id]:
                visited[neighbor_id] = True
                queue.append(neighbor_id)

    return np.flatnonzero(visited).astype(np.int64)


def keep_patch_plane_cluster_near_center(signed_depth, gap_threshold):
    signed_depth = np.asarray(signed_depth, dtype=np.float32).reshape(-1)
    if len(signed_depth) <= 1 or gap_threshold is None or float(gap_threshold) <= 0.0:
        return np.arange(len(signed_depth), dtype=np.int64)

    sort_ids = np.argsort(signed_depth)
    sorted_depth = signed_depth[sort_ids]
    anchor_sorted_idx = int(np.argmin(np.abs(sorted_depth)))

    left = anchor_sorted_idx
    while left > 0:
        if float(sorted_depth[left] - sorted_depth[left - 1]) > float(gap_threshold):
            break
        left -= 1

    right = anchor_sorted_idx
    while right < len(sorted_depth) - 1:
        if float(sorted_depth[right + 1] - sorted_depth[right]) > float(gap_threshold):
            break
        right += 1

    return np.sort(sort_ids[left : right + 1].astype(np.int64))


def keep_patch_dominant_normal_cluster(
    normal_cos,
    gap_threshold,
    min_prefix_ratio=0.25,
    min_prefix_count=6,
):
    normal_cos = np.asarray(normal_cos, dtype=np.float32).reshape(-1)
    if len(normal_cos) <= 1 or gap_threshold is None or float(gap_threshold) <= 0.0:
        return np.arange(len(normal_cos), dtype=np.int64)

    sort_ids = np.argsort(-normal_cos)
    sorted_cos = normal_cos[sort_ids]
    gaps = sorted_cos[:-1] - sorted_cos[1:]
    if len(gaps) == 0:
        return np.arange(len(normal_cos), dtype=np.int64)

    min_prefix = max(
        int(min_prefix_count),
        int(np.ceil(float(min_prefix_ratio) * float(len(sorted_cos)))),
    )
    min_prefix = min(max(2, min_prefix), len(sorted_cos) - 1)

    split_candidates = np.flatnonzero(gaps > float(gap_threshold))
    split_candidates = split_candidates[split_candidates + 1 >= min_prefix]
    if len(split_candidates) == 0:
        return np.arange(len(normal_cos), dtype=np.int64)

    split_idx = int(split_candidates[0])
    return np.sort(sort_ids[: split_idx + 1].astype(np.int64))


def filter_patch_ids(
    dense_points,
    dense_normals,
    dense_tree,
    center_point,
    center_normal,
    patch_radius,
    patch_thickness,
    min_normal_cos=0.05,
    patch_dominant_normal_gap_cos=0.18,
    patch_plane_gap_ratio=0.35,
    patch_link_radius_ratio=0.0,
    nearest_fallback_k=256,
):
    patch_ids = dense_tree.query_ball_point(center_point, r=float(patch_radius))
    if patch_ids:
        patch_ids = np.asarray(patch_ids, dtype=np.int64)
    else:
        patch_ids = np.zeros((0,), dtype=np.int64)

    if len(patch_ids) > 0 and min_normal_cos is not None:
        normal_cos = np.einsum("ij,j->i", dense_normals[patch_ids], center_normal)
        patch_ids = patch_ids[normal_cos >= float(min_normal_cos)]

    signed_depth = None
    if len(patch_ids) > 0 and patch_thickness is not None:
        offsets = dense_points[patch_ids] - center_point[None, :]
        signed_depth = np.einsum("ij,j->i", offsets, center_normal)
        keep_mask = np.abs(signed_depth) <= float(patch_thickness)
        patch_ids = patch_ids[keep_mask]
        signed_depth = signed_depth[keep_mask]

    if len(patch_ids) > 1 and patch_plane_gap_ratio is not None and patch_thickness is not None:
        plane_keep_ids = keep_patch_plane_cluster_near_center(
            signed_depth=signed_depth,
            gap_threshold=float(patch_thickness) * float(patch_plane_gap_ratio),
        )
        if len(plane_keep_ids) > 0:
            patch_ids = patch_ids[plane_keep_ids]
            signed_depth = signed_depth[plane_keep_ids]

    if len(patch_ids) > 1 and patch_dominant_normal_gap_cos is not None:
        normal_cos = np.einsum("ij,j->i", dense_normals[patch_ids], center_normal)
        dominant_keep_ids = keep_patch_dominant_normal_cluster(
            normal_cos=normal_cos,
            gap_threshold=float(patch_dominant_normal_gap_cos),
        )
        if len(dominant_keep_ids) > 0:
            patch_ids = patch_ids[dominant_keep_ids]

    if len(patch_ids) > 1 and patch_link_radius_ratio is not None:
        component_keep_ids = keep_patch_component_near_center(
            points=dense_points[patch_ids],
            center_point=center_point,
            link_radius=float(patch_radius) * float(patch_link_radius_ratio),
        )
        if len(component_keep_ids) > 0:
            patch_ids = patch_ids[component_keep_ids]

    if len(patch_ids) == 0:
        nearest_k = min(int(nearest_fallback_k), len(dense_points))
        _, nearest_ids = dense_tree.query(center_point[None, :], k=nearest_k)
        patch_ids = np.asarray(nearest_ids, dtype=np.int64).reshape(-1)
        if min_normal_cos is not None and len(patch_ids) > 0:
            normal_cos = np.einsum("ij,j->i", dense_normals[patch_ids], center_normal)
            keep_mask = normal_cos >= max(float(min_normal_cos), -0.5)
            if np.any(keep_mask):
                patch_ids = patch_ids[keep_mask]
        if len(patch_ids) > 1 and patch_dominant_normal_gap_cos is not None:
            normal_cos = np.einsum("ij,j->i", dense_normals[patch_ids], center_normal)
            dominant_keep_ids = keep_patch_dominant_normal_cluster(
                normal_cos=normal_cos,
                gap_threshold=float(patch_dominant_normal_gap_cos),
            )
            if len(dominant_keep_ids) > 0:
                patch_ids = patch_ids[dominant_keep_ids]

    return np.unique(patch_ids.astype(np.int64))


def sample_patch_points_from_ids(dense_points, patch_ids, points_per_touch, rng):
    if len(patch_ids) == 0:
        raise RuntimeError("patch_ids is empty in sample_patch_points_from_ids.")

    replace = len(patch_ids) < int(points_per_touch)
    choose_local = rng.choice(len(patch_ids), size=int(points_per_touch), replace=replace)
    return dense_points[patch_ids[choose_local]].astype(np.float32)


def sample_patch_points_and_normals_from_ids(
    dense_points,
    dense_normals,
    patch_ids,
    points_per_touch,
    rng,
):
    if len(patch_ids) == 0:
        raise RuntimeError("patch_ids is empty in sample_patch_points_and_normals_from_ids.")

    replace = len(patch_ids) < int(points_per_touch)
    choose_local = rng.choice(len(patch_ids), size=int(points_per_touch), replace=replace)
    chosen_ids = patch_ids[choose_local]
    return (
        dense_points[chosen_ids].astype(np.float32),
        dense_normals[chosen_ids].astype(np.float32),
    )


def score_candidate_coverage(
    dense_points,
    dense_normals,
    dense_tree,
    covered_mask,
    center_point,
    center_normal,
    patch_radius,
    patch_thickness,
    patch_min_normal_cos,
    patch_dominant_normal_gap_cos,
    patch_plane_gap_ratio,
    patch_link_radius_ratio,
):
    patch_ids = filter_patch_ids(
        dense_points=dense_points,
        dense_normals=dense_normals,
        dense_tree=dense_tree,
        center_point=center_point,
        center_normal=center_normal,
        patch_radius=patch_radius,
        patch_thickness=patch_thickness,
        min_normal_cos=patch_min_normal_cos,
        patch_dominant_normal_gap_cos=patch_dominant_normal_gap_cos,
        patch_plane_gap_ratio=patch_plane_gap_ratio,
        patch_link_radius_ratio=patch_link_radius_ratio,
    )
    uncovered_gain = int(np.count_nonzero(~covered_mask[patch_ids]))
    return uncovered_gain, patch_ids


def build_proposal_ids(
    candidate_points,
    candidate_tree,
    dense_points,
    covered_mask,
    used_candidate_mask,
    accepted_contact_points,
    proposal_count,
    rng,
):
    available_ids = np.flatnonzero(~used_candidate_mask)
    if len(available_ids) == 0:
        return np.zeros((0,), dtype=np.int64)

    proposal_set = set()

    random_count = min(len(available_ids), max(96, int(proposal_count) // 2))
    random_ids = rng.choice(available_ids, size=random_count, replace=False)
    proposal_set.update(np.asarray(random_ids, dtype=np.int64).tolist())

    uncovered_ids = np.flatnonzero(~covered_mask)
    if len(uncovered_ids) > 0:
        anchor_count = min(len(uncovered_ids), max(64, int(proposal_count) // 3))
        anchor_ids = rng.choice(uncovered_ids, size=anchor_count, replace=False)
        anchor_points = dense_points[anchor_ids]
        nearest_k = min(6, len(candidate_points))
        _, nearest_candidate_ids = candidate_tree.query(anchor_points, k=nearest_k)
        nearest_candidate_ids = np.asarray(nearest_candidate_ids, dtype=np.int64).reshape(-1)
        nearest_candidate_ids = nearest_candidate_ids[~used_candidate_mask[nearest_candidate_ids]]
        proposal_set.update(nearest_candidate_ids.tolist())

    if accepted_contact_points:
        accepted_points = np.asarray(accepted_contact_points, dtype=np.float32)
        probe_count = min(len(available_ids), max(96, int(proposal_count) // 3))
        probe_ids = rng.choice(available_ids, size=probe_count, replace=False)
        probe_points = candidate_points[probe_ids]
        diff = probe_points[:, None, :] - accepted_points[None, :, :]
        min_dist_sq = np.sum(diff * diff, axis=2).min(axis=1)
        far_ids = probe_ids[np.argsort(-min_dist_sq)[: min(96, len(probe_ids))]]
        proposal_set.update(np.asarray(far_ids, dtype=np.int64).tolist())

    proposal_ids = np.asarray(sorted(proposal_set), dtype=np.int64)
    if len(proposal_ids) == 0:
        return available_ids[: min(len(available_ids), int(proposal_count))]
    return proposal_ids


def candidate_payload_better(lhs, rhs):
    if rhs is None:
        return True
    if int(lhs[12]) != int(rhs[12]):
        return int(lhs[12]) > int(rhs[12])
    if float(lhs[13]) != float(rhs[13]):
        return float(lhs[13]) < float(rhs[13])
    return int(lhs[10]) > int(rhs[10])


def build_touch_view_arrays(
    accepted_patch_points,
    accepted_patch_point_normals,
    accepted_patch_centers,
    accepted_patch_center_normals,
    accepted_target_points,
    accepted_target_normals,
    accepted_contact_points,
    accepted_contact_normals,
    accepted_probe_positions,
    accepted_probe_quaternions,
    accepted_patch_source_counts,
    coverage_progress,
    num_tactile_samples,
    tactile_num_fingers,
    tactile_points_per_finger,
):
    num_views = int(num_tactile_samples)
    num_fingers = int(tactile_num_fingers)
    points_per_finger = int(tactile_points_per_finger)
    total_touches = num_views * num_fingers

    if len(accepted_patch_points) != total_touches:
        raise RuntimeError(
            f"Touch count mismatch: got {len(accepted_patch_points)}, expected {total_touches}"
        )

    view_point_count = num_fingers * points_per_finger
    touch_points = np.zeros((num_views, view_point_count, 3), dtype=np.float32)
    touch_point_normals = np.zeros((num_views, view_point_count, 3), dtype=np.float32)
    touch_round_ids = np.zeros((num_views, view_point_count), dtype=np.int32)
    touch_finger_ids = np.zeros((num_views, view_point_count), dtype=np.int32)
    touch_center_ids = np.zeros((num_views, view_point_count), dtype=np.int32)

    touch_centers = np.zeros((num_views, num_fingers, 3), dtype=np.float32)
    touch_center_normals = np.zeros((num_views, num_fingers, 3), dtype=np.float32)
    touch_target_points = np.zeros((num_views, num_fingers, 3), dtype=np.float32)
    touch_target_normals = np.zeros((num_views, num_fingers, 3), dtype=np.float32)
    touch_contact_points = np.zeros((num_views, num_fingers, 3), dtype=np.float32)
    touch_contact_normals = np.zeros((num_views, num_fingers, 3), dtype=np.float32)
    touch_probe_positions = np.zeros((num_views, num_fingers, 3), dtype=np.float32)
    touch_probe_quaternions_wxyz = np.zeros((num_views, num_fingers, 4), dtype=np.float32)
    touch_patch_source_counts = np.zeros((num_views, num_fingers), dtype=np.int32)

    for global_touch_idx in range(total_touches):
        view_idx = global_touch_idx // num_fingers
        finger_idx = global_touch_idx % num_fingers
        start = finger_idx * points_per_finger
        end = start + points_per_finger

        touch_points[view_idx, start:end] = accepted_patch_points[global_touch_idx]
        touch_point_normals[view_idx, start:end] = accepted_patch_point_normals[global_touch_idx]
        touch_finger_ids[view_idx, start:end] = finger_idx
        touch_center_ids[view_idx, start:end] = finger_idx

        touch_centers[view_idx, finger_idx] = accepted_patch_centers[global_touch_idx]
        touch_center_normals[view_idx, finger_idx] = accepted_patch_center_normals[global_touch_idx]
        touch_target_points[view_idx, finger_idx] = accepted_target_points[global_touch_idx]
        touch_target_normals[view_idx, finger_idx] = accepted_target_normals[global_touch_idx]
        touch_contact_points[view_idx, finger_idx] = accepted_contact_points[global_touch_idx]
        touch_contact_normals[view_idx, finger_idx] = accepted_contact_normals[global_touch_idx]
        touch_probe_positions[view_idx, finger_idx] = accepted_probe_positions[global_touch_idx]
        touch_probe_quaternions_wxyz[view_idx, finger_idx] = accepted_probe_quaternions[global_touch_idx]
        touch_patch_source_counts[view_idx, finger_idx] = int(accepted_patch_source_counts[global_touch_idx])

    touch_coverage_progress = np.asarray(coverage_progress, dtype=np.float32).reshape(num_views, num_fingers)
    planning_view_coverage_ratio = touch_coverage_progress[:, -1].astype(np.float32)

    return {
        "touch_points": touch_points,
        "touch_point_normals": touch_point_normals,
        "touch_round_ids": touch_round_ids,
        "touch_finger_ids": touch_finger_ids,
        "touch_center_ids": touch_center_ids,
        "touch_centers": touch_centers,
        "touch_center_normals": touch_center_normals,
        "touch_target_points": touch_target_points,
        "touch_target_normals": touch_target_normals,
        "touch_contact_points": touch_contact_points,
        "touch_contact_normals": touch_contact_normals,
        "touch_probe_positions": touch_probe_positions,
        "touch_probe_quaternions_wxyz": touch_probe_quaternions_wxyz,
        "touch_patch_source_counts": touch_patch_source_counts,
        "touch_coverage_progress": touch_coverage_progress,
        "planning_view_coverage_ratio": planning_view_coverage_ratio,
    }


def generate_mujoco_touch_data_coverage_aware(
    pipeline,
    normalized_mesh,
    normalized_mesh_path,
    num_tactile_samples=10,
    tactile_num_fingers=10,
    tactile_points_per_finger=3000,
    dense_surface_sample_n=120000,
    candidate_touch_samples=6000,
    patch_radius_ratio=0.10,
    min_touch_separation_ratio=0.055,
    patch_thickness_ratio=0.035,
    patch_min_normal_cos=0.05,
    patch_dominant_normal_gap_cos=0.18,
    patch_plane_gap_ratio=0.35,
    patch_link_radius_ratio=0.0,
    max_target_contact_offset_ratio=0.60,
    reachable_clearance_ratio=0.92,
    reachable_approach_steps=5,
    touch_mode="sphere",
    probe_geom="sphere",
    probe_radius=0.05,
    probe_capsule_half_length=0.04,
    probe_box_half_extents=None,
    approach_offset=0.18,
    indentation_depth=0.01,
    approach_steps=80,
    background_color=None,
    seed=42,
):
    if probe_box_half_extents is None:
        probe_box_half_extents = np.asarray([0.03, 0.03, 0.04], dtype=np.float32)
    else:
        probe_box_half_extents = np.asarray(probe_box_half_extents, dtype=np.float32)

    if background_color is None:
        background_color = np.asarray([0.88, 0.94, 1.0], dtype=np.float32)
    else:
        background_color = np.asarray(background_color, dtype=np.float32)

    rng = np.random.default_rng(int(seed))
    total_touches = int(num_tactile_samples) * int(tactile_num_fingers)

    raw_dense_surface_points, raw_dense_surface_normals = sample_dense_surface_points(
        normalized_mesh,
        int(dense_surface_sample_n),
    )
    diag = compute_bbox_diag(raw_dense_surface_points)

    patch_radius = float(patch_radius_ratio) * diag
    patch_thickness = float(patch_thickness_ratio) * diag
    min_touch_separation = float(min_touch_separation_ratio) * diag
    max_target_contact_offset = float(max_target_contact_offset_ratio) * patch_radius
    tactile_scene = build_raycast_scene(normalized_mesh)

    reachable_dense_mask = compute_sphere_reachability_mask(
        scene=tactile_scene,
        surface_points=raw_dense_surface_points,
        surface_normals=raw_dense_surface_normals,
        probe_radius=float(probe_radius),
        approach_offset=float(approach_offset),
        clearance_ratio=float(reachable_clearance_ratio),
        approach_steps=int(reachable_approach_steps),
    )
    reachable_dense_count = int(np.count_nonzero(reachable_dense_mask))
    minimum_dense_count = max(int(total_touches) * 32, 4096)
    if reachable_dense_count >= minimum_dense_count:
        dense_surface_points = raw_dense_surface_points[reachable_dense_mask]
        dense_surface_normals = raw_dense_surface_normals[reachable_dense_mask]
        print(
            "[INFO] tactile planning surface reachability filter kept "
            f"{reachable_dense_count}/{len(raw_dense_surface_points)} points "
            f"({reachable_dense_count / max(1, len(raw_dense_surface_points)):.4f})"
        )
    else:
        dense_surface_points = raw_dense_surface_points
        dense_surface_normals = raw_dense_surface_normals
        reachable_dense_count = int(len(raw_dense_surface_points))
        print(
            "[WARN] tactile planning reachability filter kept too few points; "
            "falling back to the unfiltered dense surface set."
        )

    dense_tree = cKDTree(dense_surface_points)

    requested_candidates = max(int(candidate_touch_samples), total_touches * 10)
    candidate_points, candidate_normals = pipeline.sample_surface_targets(
        normalized_mesh,
        num_touches=requested_candidates,
        candidate_count=requested_candidates * 2,
        seed=int(seed),
    )
    candidate_points = np.asarray(candidate_points, dtype=np.float32)
    candidate_normals = np.asarray(candidate_normals, dtype=np.float32)
    reachable_candidate_mask = compute_sphere_reachability_mask(
        scene=tactile_scene,
        surface_points=candidate_points,
        surface_normals=candidate_normals,
        probe_radius=float(probe_radius),
        approach_offset=float(approach_offset),
        clearance_ratio=float(reachable_clearance_ratio),
        approach_steps=int(reachable_approach_steps),
    )
    candidate_points = candidate_points[reachable_candidate_mask]
    candidate_normals = candidate_normals[reachable_candidate_mask]

    minimum_candidate_count = max(int(total_touches) * 8, 512)
    target_candidate_count = max(requested_candidates, minimum_candidate_count)
    if len(candidate_points) < minimum_candidate_count:
        top_up_count = target_candidate_count - len(candidate_points)
        top_up_ids = rng.choice(
            len(dense_surface_points),
            size=int(top_up_count),
            replace=len(dense_surface_points) < int(top_up_count),
        )
        candidate_points = np.concatenate(
            [candidate_points, dense_surface_points[top_up_ids].astype(np.float32)],
            axis=0,
        )
        candidate_normals = np.concatenate(
            [candidate_normals, dense_surface_normals[top_up_ids].astype(np.float32)],
            axis=0,
        )
        print(
            "[WARN] reachable candidate pool was small; topped up candidates from the "
            "reachable tactile planning surface."
        )

    candidate_tree = cKDTree(candidate_points)
    used_candidate_mask = np.zeros(len(candidate_points), dtype=bool)

    if touch_mode != "sphere":
        raise ValueError(
            "This MuJoCo batch script currently supports touch_mode='sphere' only. "
            "If you want ur5_ee / ur5_arm, I can open another version."
        )

    model, data, probe_joint_id, probe_geom_id, object_geom_id = pipeline.build_mujoco_model(
        Path(normalized_mesh_path),
        probe_geom=probe_geom,
        probe_radius=float(probe_radius),
        probe_capsule_half_length=float(probe_capsule_half_length),
        probe_box_half_extents=probe_box_half_extents,
        background_color=background_color,
    )

    covered_mask = np.zeros(len(dense_surface_points), dtype=bool)
    accepted_patch_points = []
    accepted_patch_point_normals = []
    accepted_patch_centers = []
    accepted_patch_center_normals = []
    accepted_target_points = []
    accepted_target_normals = []
    accepted_contact_points = []
    accepted_contact_normals = []
    accepted_probe_positions = []
    accepted_probe_quaternions = []
    accepted_patch_source_counts = []
    coverage_progress = []

    for touch_slot in range(total_touches):
        proposal_ids = build_proposal_ids(
            candidate_points=candidate_points,
            candidate_tree=candidate_tree,
            dense_points=dense_surface_points,
            covered_mask=covered_mask,
            used_candidate_mask=used_candidate_mask,
            accepted_contact_points=accepted_contact_points,
            proposal_count=320,
            rng=rng,
        )

        if len(proposal_ids) == 0:
            if len(candidate_points) > 0 and np.any(used_candidate_mask):
                print(
                    f"[WARN] candidate proposal pool exhausted at slot={touch_slot:03d}; "
                    "recycling proposal ids for repeat-touch fallback."
                )
                used_candidate_mask[:] = False
                proposal_ids = build_proposal_ids(
                    candidate_points=candidate_points,
                    candidate_tree=candidate_tree,
                    dense_points=dense_surface_points,
                    covered_mask=covered_mask,
                    used_candidate_mask=used_candidate_mask,
                    accepted_contact_points=accepted_contact_points,
                    proposal_count=320,
                    rng=rng,
                )

        if len(proposal_ids) == 0:
            raise RuntimeError(
                f"No candidate proposals remain before touch slot {touch_slot}."
            )

        scored_candidates = []
        for proposal_idx in proposal_ids:
            if used_candidate_mask[proposal_idx]:
                continue
            uncovered_gain, _ = score_candidate_coverage(
                dense_points=dense_surface_points,
                dense_normals=dense_surface_normals,
                dense_tree=dense_tree,
                covered_mask=covered_mask,
                center_point=candidate_points[proposal_idx],
                center_normal=normalize_vector(candidate_normals[proposal_idx]),
                patch_radius=patch_radius,
                patch_thickness=patch_thickness,
                patch_min_normal_cos=patch_min_normal_cos,
                patch_dominant_normal_gap_cos=patch_dominant_normal_gap_cos,
                patch_plane_gap_ratio=patch_plane_gap_ratio,
                patch_link_radius_ratio=patch_link_radius_ratio,
            )
            scored_candidates.append((uncovered_gain, int(proposal_idx)))

        scored_candidates.sort(reverse=True)
        attempt_order = [idx for _, idx in scored_candidates[:96]]

        if len(attempt_order) == 0:
            available_ids = np.flatnonzero(~used_candidate_mask)
            attempt_order = available_ids[: min(len(available_ids), 96)].tolist()

        accepted_this_touch = False
        best_repeat_candidate = None
        best_relaxed_candidate = None
        best_separation_relaxed_candidate = None

        for proposal_idx in attempt_order:
            if used_candidate_mask[proposal_idx]:
                continue
            used_candidate_mask[proposal_idx] = True

            target_point = candidate_points[proposal_idx]
            target_normal = normalize_vector(candidate_normals[proposal_idx])

            contact_result = pipeline.simulate_touch_contact(
                model=model,
                data=data,
                probe_joint_id=probe_joint_id,
                probe_geom_id=probe_geom_id,
                object_geom_id=object_geom_id,
                target_point=target_point,
                outward_normal=target_normal,
                touch_mode=touch_mode,
                probe_geom=probe_geom,
                probe_radius=float(probe_radius),
                probe_capsule_half_length=float(probe_capsule_half_length),
                probe_box_half_extents=probe_box_half_extents,
                approach_offset=float(approach_offset),
                indentation_depth=float(indentation_depth),
                approach_steps=int(approach_steps),
                ur5_roll_jitter_deg=0.0,
                rng=rng,
                viewer=None,
                viewer_sleep=0.0,
            )
            if contact_result is None:
                continue

            contact_point, contact_normal, probe_position, probe_quaternion = contact_result
            contact_point = np.asarray(contact_point, dtype=np.float32)
            contact_normal = align_normal_to_reference(contact_normal, target_normal)
            probe_position = np.asarray(probe_position, dtype=np.float32)
            probe_quaternion = np.asarray(probe_quaternion, dtype=np.float32)

            target_contact_offset = float(
                np.linalg.norm(contact_point.astype(np.float32) - target_point.astype(np.float32))
            )
            patch_center_point = target_point.astype(np.float32)
            patch_center_normal = target_normal.astype(np.float32)
            patch_ids = filter_patch_ids(
                dense_points=dense_surface_points,
                dense_normals=dense_surface_normals,
                dense_tree=dense_tree,
                center_point=patch_center_point,
                center_normal=patch_center_normal,
                patch_radius=float(patch_radius),
                patch_thickness=float(patch_thickness),
                min_normal_cos=patch_min_normal_cos,
                patch_dominant_normal_gap_cos=patch_dominant_normal_gap_cos,
                patch_plane_gap_ratio=patch_plane_gap_ratio,
                patch_link_radius_ratio=patch_link_radius_ratio,
            )
            patch_points, patch_point_normals = sample_patch_points_and_normals_from_ids(
                dense_points=dense_surface_points,
                dense_normals=dense_surface_normals,
                patch_ids=patch_ids,
                points_per_touch=int(tactile_points_per_finger),
                rng=rng,
            )
            source_count = int(len(patch_ids))
            uncovered_gain = int(np.count_nonzero(~covered_mask[patch_ids]))

            candidate_payload = (
                patch_points.astype(np.float32),
                patch_point_normals.astype(np.float32),
                patch_center_point.astype(np.float32),
                patch_center_normal.astype(np.float32),
                target_point.astype(np.float32),
                target_normal.astype(np.float32),
                contact_point.astype(np.float32),
                contact_normal.astype(np.float32),
                probe_position.astype(np.float32),
                probe_quaternion.astype(np.float32),
                int(source_count),
                np.asarray(patch_ids, dtype=np.int64),
                uncovered_gain,
                float(target_contact_offset),
            )

            if accepted_contact_points:
                min_sep = np.min(
                    np.linalg.norm(
                        np.asarray(accepted_contact_points, dtype=np.float32) - contact_point[None, :],
                        axis=1,
                    )
                )
                if min_sep < float(min_touch_separation):
                    if candidate_payload_better(candidate_payload, best_separation_relaxed_candidate):
                        best_separation_relaxed_candidate = candidate_payload
                    continue

            if target_contact_offset > float(max_target_contact_offset):
                if candidate_payload_better(candidate_payload, best_relaxed_candidate):
                    best_relaxed_candidate = candidate_payload
                continue

            if uncovered_gain > 0:
                (
                    patch_points,
                    patch_point_normals,
                    patch_center_point,
                    patch_center_normal,
                    target_point,
                    target_normal,
                    contact_point,
                    contact_normal,
                    probe_position,
                    probe_quaternion,
                    source_count,
                    patch_ids,
                    uncovered_gain,
                    target_contact_offset,
                ) = candidate_payload
                accepted_this_touch = True
                break

            if candidate_payload_better(candidate_payload, best_repeat_candidate):
                best_repeat_candidate = candidate_payload

        if not accepted_this_touch:
            fallback_reason = None
            if best_repeat_candidate is not None:
                fallback_reason = "repeat"
                fallback_candidate = best_repeat_candidate
            elif best_relaxed_candidate is not None:
                fallback_reason = "relaxed-offset"
                fallback_candidate = best_relaxed_candidate
            elif best_separation_relaxed_candidate is not None:
                fallback_reason = "relaxed-separation"
                fallback_candidate = best_separation_relaxed_candidate
            else:
                raise RuntimeError(
                    f"Failed to simulate a valid MuJoCo touch for slot {touch_slot}."
                )
            (
                patch_points,
                patch_point_normals,
                patch_center_point,
                patch_center_normal,
                target_point,
                target_normal,
                contact_point,
                contact_normal,
                probe_position,
                probe_quaternion,
                source_count,
                patch_ids,
                uncovered_gain,
                target_contact_offset,
            ) = fallback_candidate
            print(
                f"[WARN] using {fallback_reason} fallback at slot={touch_slot:03d} "
                f"offset={float(target_contact_offset):.6f}"
            )

        accepted_patch_points.append(patch_points)
        accepted_patch_point_normals.append(patch_point_normals)
        accepted_patch_centers.append(patch_center_point)
        accepted_patch_center_normals.append(patch_center_normal)
        accepted_target_points.append(target_point)
        accepted_target_normals.append(target_normal)
        accepted_contact_points.append(contact_point)
        accepted_contact_normals.append(contact_normal)
        accepted_probe_positions.append(probe_position)
        accepted_probe_quaternions.append(probe_quaternion)
        accepted_patch_source_counts.append(int(source_count))

        covered_mask[patch_ids] = True
        current_coverage = float(np.mean(covered_mask))
        coverage_progress.append(current_coverage)

        print(
            f"[TOUCH mujoco-coverage] slot={touch_slot:03d} "
            f"new_cover={uncovered_gain:06d} coverage={current_coverage:.4f} "
            f"source_points={int(source_count)} "
            f"target_contact_offset={float(target_contact_offset):.6f}"
        )

    touch_data = build_touch_view_arrays(
        accepted_patch_points=accepted_patch_points,
        accepted_patch_point_normals=accepted_patch_point_normals,
        accepted_patch_centers=accepted_patch_centers,
        accepted_patch_center_normals=accepted_patch_center_normals,
        accepted_target_points=accepted_target_points,
        accepted_target_normals=accepted_target_normals,
        accepted_contact_points=accepted_contact_points,
        accepted_contact_normals=accepted_contact_normals,
        accepted_probe_positions=accepted_probe_positions,
        accepted_probe_quaternions=accepted_probe_quaternions,
        accepted_patch_source_counts=accepted_patch_source_counts,
        coverage_progress=coverage_progress,
        num_tactile_samples=num_tactile_samples,
        tactile_num_fingers=tactile_num_fingers,
        tactile_points_per_finger=tactile_points_per_finger,
    )
    touch_data["planning_surface_coverage_ratio"] = np.array(
        coverage_progress[-1] if coverage_progress else 0.0,
        dtype=np.float32,
    )
    touch_data["planning_dense_surface_point_count"] = np.array(
        len(raw_dense_surface_points),
        dtype=np.int32,
    )
    touch_data["planning_reachable_surface_point_count"] = np.array(
        len(dense_surface_points),
        dtype=np.int32,
    )
    touch_data["planning_reachable_surface_fraction"] = np.array(
        float(len(dense_surface_points)) / float(max(1, len(raw_dense_surface_points))),
        dtype=np.float32,
    )
    touch_data["planning_candidate_point_count"] = np.array(
        len(candidate_points),
        dtype=np.int32,
    )
    return touch_data


def process_single_obj_to_mujoco_coverage_npz(
    obj_path,
    out_path,
    normalized_mesh_asset_path,
    num_tactile_samples=10,
    tactile_num_fingers=10,
    tactile_points_per_finger=3000,
    dense_surface_sample_n=120000,
    candidate_touch_samples=6000,
    tactile_patch_radius_ratio=0.10,
    tactile_min_touch_separation_ratio=0.055,
    tactile_patch_thickness_ratio=0.035,
    patch_min_normal_cos=0.05,
    tactile_patch_dominant_normal_gap_cos=0.18,
    tactile_patch_plane_gap_ratio=0.35,
    tactile_patch_link_radius_ratio=0.0,
    max_target_contact_offset_ratio=0.60,
    tactile_reachable_clearance_ratio=0.92,
    tactile_reachable_approach_steps=5,
    normalization_bound=0.9,
    num_surface_points=235000,
    num_query_points=250000,
    query_uniform_region="cube",
    query_sampling_mode="paired_normal_offsets",
    paired_query_fraction=0.90,
    paired_query_eps_min=0.002,
    paired_query_eps_max=0.025,
    paired_query_max_attempts=8,
    paired_query_anchor_mode="coverage_grid",
    paired_query_coverage_grid_size=12,
    paired_query_coverage_min_per_cell=1,
    paired_query_eps_retries=3,
    watertight_proxy_mode="repair",
    watertight_mesh_usage="sign_proxy",
    non_watertight_policy="skip",
    proxy_poisson_samples=50000,
    proxy_poisson_depth=8,
    proxy_poisson_full_depth=5,
    proxy_poisson_threads=8,
    manifoldplus_path=None,
    manifoldplus_depth=8,
    mujoco_max_faces=190000,
    query_occupancy_nsamples=11,
    query_near_surface_sign_band=0.01,
    seed=42,
):
    print("\n==================================================")
    print("[PROCESS mujoco coverage-aware]", obj_path)
    print("==================================================")

    pipeline = load_tactistruct_pipeline_module()
    mesh_name = os.path.splitext(os.path.basename(obj_path))[0]

    source_mesh = pipeline.load_input_mesh(Path(obj_path))
    normalized_mesh, transform = pipeline.normalize_mesh(
        source_mesh,
        float(normalization_bound),
    )

    watertight_mesh_usage = str(watertight_mesh_usage).strip().lower()
    mesh_geometry_source = "original_normalized"
    surface_points, surface_normals = sample_surface_points_for_storage(
        normalized_mesh,
        num_surface_points=num_surface_points,
    )
    surface_normals = orient_normals_outward_from_center(
        surface_points,
        surface_normals,
        normalized_mesh.bounding_box.centroid,
    )

    if watertight_mesh_usage == "full_pipeline":
        print(
            f"[INFO] building watertight mesh before downstream sampling "
            f"(mode={watertight_proxy_mode}) ..."
        )
        watertight_mesh, watertight_source = build_sign_proxy_mesh(
            mesh=normalized_mesh,
            surface_points=surface_points,
            surface_normals=surface_normals,
            mode=watertight_proxy_mode,
            poisson_sample_count=proxy_poisson_samples,
            poisson_depth=proxy_poisson_depth,
            poisson_full_depth=proxy_poisson_full_depth,
            poisson_threads=proxy_poisson_threads,
            manifoldplus_path=manifoldplus_path,
            manifoldplus_depth=manifoldplus_depth,
            seed=int(seed) + 104729,
        )
        if watertight_mesh is not None and watertight_mesh.is_watertight:
            normalized_mesh = watertight_mesh
            mesh_geometry_source = f"watertight_{watertight_source}"
            print(
                f"[INFO] full pipeline mesh replaced by {mesh_geometry_source} "
                f"(vertices={len(normalized_mesh.vertices)}, faces={len(normalized_mesh.faces)})"
            )
            surface_points, surface_normals = sample_surface_points_for_storage(
                normalized_mesh,
                num_surface_points=num_surface_points,
            )
            surface_normals = orient_normals_outward_from_center(
                surface_points,
                surface_normals,
                normalized_mesh.bounding_box.centroid,
            )
        elif str(non_watertight_policy).strip().lower() == "skip":
            raise RuntimeError(
                "No watertight full-pipeline mesh could be built and "
                "--non-watertight-policy=skip was selected."
            )
        else:
            print(
                "[WARN] no watertight full-pipeline mesh was built; continuing with "
                "the original normalized mesh because fallback was selected."
            )
    elif watertight_mesh_usage != "sign_proxy":
        raise ValueError(
            f"Unsupported watertight_mesh_usage={watertight_mesh_usage!r}. "
            "Expected 'sign_proxy' or 'full_pipeline'."
        )

    normalized_mesh_asset_path = Path(normalized_mesh_asset_path)
    normalized_mesh_asset_path.parent.mkdir(parents=True, exist_ok=True)
    mujoco_mesh, mujoco_mesh_was_simplified = simplify_mesh_for_mujoco(
        normalized_mesh,
        max_faces=mujoco_max_faces,
    )
    if mujoco_mesh_was_simplified:
        print(
            f"[INFO] MuJoCo export mesh decimated from {len(normalized_mesh.faces)} "
            f"to {len(mujoco_mesh.faces)} faces."
        )
    mujoco_mesh.export(normalized_mesh_asset_path)

    scene = build_raycast_scene(normalized_mesh)
    query_rng = np.random.default_rng(int(seed) + 7919)
    query_sampling_mode = str(query_sampling_mode).strip().lower()

    query_pair_ids = np.full((int(num_query_points),), -1, dtype=np.int32)
    query_pair_side = np.zeros((int(num_query_points),), dtype=np.int8)
    query_anchor_ids = np.full((int(num_query_points),), -1, dtype=np.int32)
    query_pair_eps = np.zeros((int(num_query_points),), dtype=np.float32)
    query_pair_metadata = {
        "requested_pairs": 0,
        "accepted_pairs": 0,
        "rejected_pairs": 0,
        "flipped_pairs": 0,
        "proxy_tested_pairs": 0,
        "fallback_pairs": 0,
        "uniform_count": int(num_query_points),
        "validated_with_proxy": False,
        "anchor_mode": "none",
        "anchor_strata_count": 0,
        "eps_retries": 0,
    }
    query_sign_metadata = {
        "sign_source": "legacy",
        "inside_count": 0,
        "outside_count": int(num_query_points),
    }
    sign_proxy_source = "legacy_original_mesh"
    sign_proxy_is_watertight = bool(normalized_mesh.is_watertight)

    if query_sampling_mode == "legacy_gaussian":
        print(
            f"[INFO] sampling legacy Gaussian query points "
            f"(uniform_region={query_uniform_region}) ..."
        )
        query_points = sample_query_points_near_surface_legacy(
            surface_points=surface_points,
            number_of_points=num_query_points,
            uniform_region=query_uniform_region,
        )

        if normalized_mesh.is_watertight:
            print("[INFO] computing legacy query_sdf using occupancy sign (watertight mesh) ...")
        else:
            print("[INFO] computing legacy query_sdf using local near-surface fallback ...")

        query_sdf = compute_query_sdf_with_raycasting_legacy(
            scene=scene,
            query_points=query_points,
            mesh_is_watertight=normalized_mesh.is_watertight,
            surface_points=surface_points,
            surface_normals=surface_normals,
            occupancy_nsamples=int(query_occupancy_nsamples),
            near_surface_sign_band=float(query_near_surface_sign_band),
        )
        query_sign_metadata = {
            "sign_source": "legacy_raycasting",
            "inside_count": int(np.count_nonzero(query_sdf < 0.0)),
            "outside_count": int(np.count_nonzero(query_sdf >= 0.0)),
        }
    elif query_sampling_mode == "paired_normal_offsets":
        print(
            f"[INFO] building sign proxy for paired query labels "
            f"(mode={watertight_proxy_mode}) ..."
        )
        sign_proxy_mesh, sign_proxy_source = build_sign_proxy_mesh(
            mesh=normalized_mesh,
            surface_points=surface_points,
            surface_normals=surface_normals,
            mode=watertight_proxy_mode,
            poisson_sample_count=proxy_poisson_samples,
            poisson_depth=proxy_poisson_depth,
            poisson_full_depth=proxy_poisson_full_depth,
            poisson_threads=proxy_poisson_threads,
            manifoldplus_path=manifoldplus_path,
            manifoldplus_depth=manifoldplus_depth,
            seed=int(seed) + 104729,
        )
        sign_proxy_is_watertight = bool(
            sign_proxy_mesh is not None and sign_proxy_mesh.is_watertight
        )
        if sign_proxy_is_watertight:
            print(f"[INFO] sign proxy ready: {sign_proxy_source} (watertight=True)")
            sign_scene = build_raycast_scene(sign_proxy_mesh)
        else:
            if str(non_watertight_policy).strip().lower() == "skip":
                raise RuntimeError(
                    "No watertight sign proxy could be built and "
                    "--non-watertight-policy=skip was selected."
                )
            print(
                "[WARN] no watertight sign proxy available; paired normal labels "
                "will be used for near-surface pairs."
            )
            sign_scene = None

        print(
            f"[INFO] sampling paired normal query points "
            f"(paired_fraction={float(paired_query_fraction):.3f}, "
            f"eps=[{float(paired_query_eps_min):.5f}, {float(paired_query_eps_max):.5f}]) ..."
        )
        (
            query_points,
            query_pair_ids,
            query_pair_side,
            query_anchor_ids,
            query_pair_eps,
            query_pair_metadata,
        ) = sample_paired_normal_query_points(
            surface_points=surface_points,
            surface_normals=surface_normals,
            number_of_points=num_query_points,
            uniform_region=query_uniform_region,
            paired_fraction=paired_query_fraction,
            eps_min=paired_query_eps_min,
            eps_max=paired_query_eps_max,
            rng=query_rng,
            sign_scene=sign_scene,
            sign_proxy_is_watertight=sign_proxy_is_watertight,
            occupancy_nsamples=query_occupancy_nsamples,
            max_attempts=paired_query_max_attempts,
            anchor_mode=paired_query_anchor_mode,
            coverage_grid_size=paired_query_coverage_grid_size,
            coverage_min_per_cell=paired_query_coverage_min_per_cell,
            eps_retries=paired_query_eps_retries,
        )
        query_sdf, query_sign_metadata = assign_query_sdf_signs(
            distance_scene=scene,
            sign_scene=sign_scene,
            query_points=query_points,
            query_pair_side=query_pair_side,
            sign_proxy_is_watertight=sign_proxy_is_watertight,
            occupancy_nsamples=query_occupancy_nsamples,
            surface_points=surface_points,
            surface_normals=surface_normals,
            near_surface_sign_band=query_near_surface_sign_band,
        )
    else:
        raise ValueError(
            f"Unsupported query_sampling_mode={query_sampling_mode!r}. "
            "Expected 'paired_normal_offsets' or 'legacy_gaussian'."
        )

    touch_data = generate_mujoco_touch_data_coverage_aware(
        pipeline=pipeline,
        normalized_mesh=normalized_mesh,
        normalized_mesh_path=normalized_mesh_asset_path,
        num_tactile_samples=num_tactile_samples,
        tactile_num_fingers=tactile_num_fingers,
        tactile_points_per_finger=tactile_points_per_finger,
        dense_surface_sample_n=dense_surface_sample_n,
        candidate_touch_samples=candidate_touch_samples,
        patch_radius_ratio=tactile_patch_radius_ratio,
        min_touch_separation_ratio=tactile_min_touch_separation_ratio,
        patch_thickness_ratio=tactile_patch_thickness_ratio,
        patch_min_normal_cos=patch_min_normal_cos,
        patch_dominant_normal_gap_cos=tactile_patch_dominant_normal_gap_cos,
        patch_plane_gap_ratio=tactile_patch_plane_gap_ratio,
        patch_link_radius_ratio=tactile_patch_link_radius_ratio,
        max_target_contact_offset_ratio=max_target_contact_offset_ratio,
        reachable_clearance_ratio=tactile_reachable_clearance_ratio,
        reachable_approach_steps=tactile_reachable_approach_steps,
        touch_mode="sphere",
        probe_geom="sphere",
        probe_radius=0.05,
        probe_capsule_half_length=0.04,
        probe_box_half_extents=np.asarray([0.03, 0.03, 0.04], dtype=np.float32),
        approach_offset=0.18,
        indentation_depth=0.01,
        approach_steps=80,
        background_color=np.asarray([0.88, 0.94, 1.0], dtype=np.float32),
        seed=int(seed),
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    np.savez_compressed(
        out_path,
        surface_points=surface_points,
        surface_normals=surface_normals,
        query_points=query_points.astype(np.float32),
        query_sdf=query_sdf.astype(np.float32),
        query_pair_ids=query_pair_ids.astype(np.int32),
        query_pair_side=query_pair_side.astype(np.int8),
        query_anchor_ids=query_anchor_ids.astype(np.int32),
        query_pair_eps=query_pair_eps.astype(np.float32),
        touch_points=touch_data["touch_points"],
        touch_point_normals=touch_data["touch_point_normals"],
        touch_round_ids=touch_data["touch_round_ids"],
        touch_finger_ids=touch_data["touch_finger_ids"],
        touch_center_ids=touch_data["touch_center_ids"],
        touch_centers=touch_data["touch_centers"],
        touch_center_normals=touch_data["touch_center_normals"],
        touch_target_points=touch_data["touch_target_points"],
        touch_target_normals=touch_data["touch_target_normals"],
        touch_contact_points=touch_data["touch_contact_points"],
        touch_contact_normals=touch_data["touch_contact_normals"],
        touch_probe_positions=touch_data["touch_probe_positions"],
        touch_probe_quaternions_wxyz=touch_data["touch_probe_quaternions_wxyz"],
        touch_patch_source_counts=touch_data["touch_patch_source_counts"],
        touch_coverage_progress=touch_data["touch_coverage_progress"],
        planning_surface_coverage_ratio=touch_data["planning_surface_coverage_ratio"],
        planning_view_coverage_ratio=touch_data["planning_view_coverage_ratio"],
        planning_dense_surface_point_count=touch_data["planning_dense_surface_point_count"],
        planning_reachable_surface_point_count=touch_data["planning_reachable_surface_point_count"],
        planning_reachable_surface_fraction=touch_data["planning_reachable_surface_fraction"],
        planning_candidate_point_count=touch_data["planning_candidate_point_count"],
        object_center=transform.center.astype(np.float32),
        object_scale=np.asarray(transform.scale, dtype=np.float32),
        normalization_bound=np.asarray(transform.target_bound, dtype=np.float32),
        source_mesh=np.asarray(str(obj_path)),
        normalized_mesh_asset=np.asarray(str(normalized_mesh_asset_path)),
        mesh_name=np.array(mesh_name),
        num_tactile_samples=np.array(num_tactile_samples, dtype=np.int32),
        tactile_num_fingers=np.array(tactile_num_fingers, dtype=np.int32),
        query_uniform_region=np.asarray(str(query_uniform_region)),
        query_sampling_mode=np.asarray(str(query_sampling_mode)),
        query_sign_convention=np.asarray("outside_positive_inside_negative"),
        watertight_mesh_usage=np.asarray(str(watertight_mesh_usage)),
        mesh_geometry_source=np.asarray(str(mesh_geometry_source)),
        mesh_geometry_is_watertight=np.asarray(bool(normalized_mesh.is_watertight)),
        mujoco_mesh_faces=np.asarray(int(len(mujoco_mesh.faces)), dtype=np.int32),
        mujoco_mesh_was_simplified=np.asarray(bool(mujoco_mesh_was_simplified)),
        query_sign_source=np.asarray(str(query_sign_metadata["sign_source"])),
        query_inside_count=np.asarray(int(query_sign_metadata["inside_count"]), dtype=np.int32),
        query_outside_count=np.asarray(int(query_sign_metadata["outside_count"]), dtype=np.int32),
        query_proxy_mode=np.asarray(str(watertight_proxy_mode)),
        query_proxy_source=np.asarray(str(sign_proxy_source)),
        query_proxy_is_watertight=np.asarray(bool(sign_proxy_is_watertight)),
        query_requested_pairs=np.asarray(int(query_pair_metadata["requested_pairs"]), dtype=np.int32),
        query_accepted_pairs=np.asarray(int(query_pair_metadata["accepted_pairs"]), dtype=np.int32),
        query_rejected_pairs=np.asarray(int(query_pair_metadata["rejected_pairs"]), dtype=np.int32),
        query_flipped_pairs=np.asarray(int(query_pair_metadata["flipped_pairs"]), dtype=np.int32),
        query_proxy_tested_pairs=np.asarray(int(query_pair_metadata["proxy_tested_pairs"]), dtype=np.int32),
        query_fallback_pairs=np.asarray(int(query_pair_metadata["fallback_pairs"]), dtype=np.int32),
        query_uniform_count=np.asarray(int(query_pair_metadata["uniform_count"]), dtype=np.int32),
        query_validated_with_proxy=np.asarray(bool(query_pair_metadata["validated_with_proxy"])),
        query_anchor_mode=np.asarray(str(query_pair_metadata["anchor_mode"])),
        query_anchor_strata_count=np.asarray(int(query_pair_metadata["anchor_strata_count"]), dtype=np.int32),
        query_eps_retries=np.asarray(int(query_pair_metadata["eps_retries"]), dtype=np.int32),
    )

    print("[SAVED]", out_path)
    print("surface_points                 :", surface_points.shape)
    print("query_points                   :", query_points.shape)
    print("query_sdf                      :", query_sdf.shape)
    print("query_sampling_mode            :", query_sampling_mode)
    print("watertight_mesh_usage          :", watertight_mesh_usage)
    print("mesh_geometry_source           :", mesh_geometry_source)
    print("mesh_geometry_is_watertight    :", normalized_mesh.is_watertight)
    print("mujoco_mesh_faces              :", len(mujoco_mesh.faces))
    print("mujoco_mesh_was_simplified     :", mujoco_mesh_was_simplified)
    print("query_sign_source              :", query_sign_metadata["sign_source"])
    print("query_proxy_source             :", sign_proxy_source)
    print("query_proxy_is_watertight      :", sign_proxy_is_watertight)
    print(
        "query inside/outside           : "
        f"{int(query_sign_metadata['inside_count'])}/"
        f"{int(query_sign_metadata['outside_count'])}"
    )
    print(
        "query pairs accepted/rejected  : "
        f"{int(query_pair_metadata['accepted_pairs'])}/"
        f"{int(query_pair_metadata['rejected_pairs'])}"
    )
    print(
        "query anchor mode/strata       : "
        f"{query_pair_metadata['anchor_mode']}/"
        f"{int(query_pair_metadata['anchor_strata_count'])}"
    )
    print(
        "query flipped/fallback pairs   : "
        f"{int(query_pair_metadata['flipped_pairs'])}/"
        f"{int(query_pair_metadata['fallback_pairs'])}"
    )
    print("touch_points                   :", touch_data["touch_points"].shape)
    print("touch_point_normals            :", touch_data["touch_point_normals"].shape)
    print("touch_centers                  :", touch_data["touch_centers"].shape)
    print(
        f"planning_surface_coverage_ratio: "
        f"{float(touch_data['planning_surface_coverage_ratio']):.4f}"
    )
    print(
        "planning_reachable_surface_fraction: "
        f"{float(touch_data['planning_reachable_surface_fraction']):.4f}"
    )

    return {
        "out_path": str(out_path),
        "normalized_mesh_asset_path": str(normalized_mesh_asset_path),
        "planning_surface_coverage_ratio": float(
            touch_data["planning_surface_coverage_ratio"]
        ),
        "planning_reachable_surface_fraction": float(
            touch_data["planning_reachable_surface_fraction"]
        ),
        "num_touch_views": int(touch_data["touch_points"].shape[0]),
        "query_proxy_is_watertight": bool(sign_proxy_is_watertight),
        "query_proxy_source": str(sign_proxy_source),
        "query_sign_source": str(query_sign_metadata["sign_source"]),
        "watertight_mesh_usage": str(watertight_mesh_usage),
        "mesh_geometry_source": str(mesh_geometry_source),
        "mesh_geometry_is_watertight": bool(normalized_mesh.is_watertight),
        "mujoco_mesh_faces": int(len(mujoco_mesh.faces)),
        "mujoco_mesh_was_simplified": bool(mujoco_mesh_was_simplified),
        "query_inside_count": int(query_sign_metadata["inside_count"]),
        "query_outside_count": int(query_sign_metadata["outside_count"]),
    }


def process_single_obj_job(job):
    obj_path = str(job["obj_path"])
    out_path = str(job["out_path"])
    normalized_mesh_asset_path = str(job["normalized_mesh_asset_path"])

    if os.path.exists(out_path) and not bool(job["overwrite"]):
        return {
            "status": "skipped",
            "obj_path": obj_path,
            "out_path": out_path,
            "message": "exists",
        }

    try:
        result = process_single_obj_to_mujoco_coverage_npz(
            obj_path=obj_path,
            out_path=out_path,
            normalized_mesh_asset_path=normalized_mesh_asset_path,
            num_tactile_samples=int(job["num_tactile_samples"]),
            tactile_num_fingers=int(job["tactile_num_fingers"]),
            tactile_points_per_finger=int(job["tactile_points_per_finger"]),
            dense_surface_sample_n=int(job["dense_surface_sample_n"]),
            candidate_touch_samples=int(job["candidate_touch_samples"]),
            tactile_patch_radius_ratio=float(job["tactile_patch_radius_ratio"]),
            tactile_min_touch_separation_ratio=float(
                job["tactile_min_touch_separation_ratio"]
            ),
            tactile_patch_thickness_ratio=float(job["tactile_patch_thickness_ratio"]),
            patch_min_normal_cos=float(job["patch_min_normal_cos"]),
            tactile_patch_dominant_normal_gap_cos=float(job["tactile_patch_dominant_normal_gap_cos"]),
            tactile_patch_plane_gap_ratio=float(job["tactile_patch_plane_gap_ratio"]),
            tactile_patch_link_radius_ratio=float(job["tactile_patch_link_radius_ratio"]),
            max_target_contact_offset_ratio=float(job["max_target_contact_offset_ratio"]),
            tactile_reachable_clearance_ratio=float(job["tactile_reachable_clearance_ratio"]),
            tactile_reachable_approach_steps=int(job["tactile_reachable_approach_steps"]),
            normalization_bound=float(job["normalization_bound"]),
            num_surface_points=int(job["num_surface_points"]),
            num_query_points=int(job["num_query_points"]),
            query_uniform_region=str(job["query_uniform_region"]),
            query_sampling_mode=str(job["query_sampling_mode"]),
            paired_query_fraction=float(job["paired_query_fraction"]),
            paired_query_eps_min=float(job["paired_query_eps_min"]),
            paired_query_eps_max=float(job["paired_query_eps_max"]),
            paired_query_max_attempts=int(job["paired_query_max_attempts"]),
            paired_query_anchor_mode=str(job["paired_query_anchor_mode"]),
            paired_query_coverage_grid_size=int(job["paired_query_coverage_grid_size"]),
            paired_query_coverage_min_per_cell=int(job["paired_query_coverage_min_per_cell"]),
            paired_query_eps_retries=int(job["paired_query_eps_retries"]),
            watertight_proxy_mode=str(job["watertight_proxy_mode"]),
            watertight_mesh_usage=str(job["watertight_mesh_usage"]),
            non_watertight_policy=str(job["non_watertight_policy"]),
            proxy_poisson_samples=int(job["proxy_poisson_samples"]),
            proxy_poisson_depth=int(job["proxy_poisson_depth"]),
            proxy_poisson_full_depth=int(job["proxy_poisson_full_depth"]),
            proxy_poisson_threads=int(job["proxy_poisson_threads"]),
            manifoldplus_path=job.get("manifoldplus_path"),
            manifoldplus_depth=int(job["manifoldplus_depth"]),
            mujoco_max_faces=int(job["mujoco_max_faces"]),
            query_occupancy_nsamples=int(job["query_occupancy_nsamples"]),
            query_near_surface_sign_band=float(job["query_near_surface_sign_band"]),
            seed=int(job["seed"]),
        )
        return {
            "status": "ok",
            "obj_path": obj_path,
            "out_path": out_path,
            "coverage": float(result["planning_surface_coverage_ratio"]),
            "reachable_fraction": float(result["planning_reachable_surface_fraction"]),
            "num_touch_views": int(result["num_touch_views"]),
            "query_proxy_is_watertight": bool(result["query_proxy_is_watertight"]),
            "query_proxy_source": str(result["query_proxy_source"]),
            "query_sign_source": str(result["query_sign_source"]),
            "watertight_mesh_usage": str(result["watertight_mesh_usage"]),
            "mesh_geometry_source": str(result["mesh_geometry_source"]),
            "mesh_geometry_is_watertight": bool(result["mesh_geometry_is_watertight"]),
            "mujoco_mesh_faces": int(result["mujoco_mesh_faces"]),
            "mujoco_mesh_was_simplified": bool(result["mujoco_mesh_was_simplified"]),
            "query_inside_count": int(result["query_inside_count"]),
            "query_outside_count": int(result["query_outside_count"]),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "obj_path": obj_path,
            "out_path": out_path,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def build_jobs(
    root_dir,
    category_names=None,
    max_objects_per_category=None,
    num_tactile_samples=10,
    tactile_num_fingers=10,
    tactile_points_per_finger=3000,
    dense_surface_sample_n=120000,
    candidate_touch_samples=6000,
    tactile_patch_radius_ratio=0.10,
    tactile_min_touch_separation_ratio=0.055,
    tactile_patch_thickness_ratio=0.035,
    patch_min_normal_cos=0.05,
    tactile_patch_dominant_normal_gap_cos=0.18,
    tactile_patch_plane_gap_ratio=0.35,
    tactile_patch_link_radius_ratio=0.0,
    max_target_contact_offset_ratio=0.60,
    tactile_reachable_clearance_ratio=0.92,
    tactile_reachable_approach_steps=5,
    normalization_bound=0.9,
    num_surface_points=235000,
    num_query_points=250000,
    query_uniform_region="cube",
    query_sampling_mode="paired_normal_offsets",
    paired_query_fraction=0.90,
    paired_query_eps_min=0.002,
    paired_query_eps_max=0.025,
    paired_query_max_attempts=8,
    paired_query_anchor_mode="coverage_grid",
    paired_query_coverage_grid_size=12,
    paired_query_coverage_min_per_cell=1,
    paired_query_eps_retries=3,
    watertight_proxy_mode="repair",
    watertight_mesh_usage="sign_proxy",
    non_watertight_policy="skip",
    proxy_poisson_samples=50000,
    proxy_poisson_depth=8,
    proxy_poisson_full_depth=5,
    proxy_poisson_threads=8,
    manifoldplus_path=None,
    manifoldplus_depth=8,
    mujoco_max_faces=190000,
    query_occupancy_nsamples=11,
    query_near_surface_sign_band=0.01,
    output_folder_name="tactistruct_npz_shapenet_mujoco_coverage_onefolder_paired_watertight_strict",
    asset_folder_name="tactistruct_npz_shapenet_mujoco_coverage_assets_paired_watertight_strict",
    overwrite=False,
    base_seed=42,
):
    category_dirs = list(iter_category_dirs(root_dir, category_names=category_names))
    jobs = []

    for category_dir in category_dirs:
        obj_paths = find_shapenet_obj_files(
            category_dir,
            max_objects=max_objects_per_category,
        )
        for obj_path in obj_paths:
            out_path = build_flat_output_path(
                obj_path=obj_path,
                root_dir=root_dir,
                output_folder_name=output_folder_name,
            )
            normalized_mesh_asset_path = build_asset_export_path(
                obj_path=obj_path,
                root_dir=root_dir,
                asset_folder_name=asset_folder_name,
            )
            jobs.append(
                {
                    "obj_path": obj_path,
                    "out_path": out_path,
                    "normalized_mesh_asset_path": normalized_mesh_asset_path,
                    "overwrite": bool(overwrite),
                    "num_tactile_samples": int(num_tactile_samples),
                    "tactile_num_fingers": int(tactile_num_fingers),
                    "tactile_points_per_finger": int(tactile_points_per_finger),
                    "dense_surface_sample_n": int(dense_surface_sample_n),
                    "candidate_touch_samples": int(candidate_touch_samples),
                    "tactile_patch_radius_ratio": float(tactile_patch_radius_ratio),
                    "tactile_min_touch_separation_ratio": float(
                        tactile_min_touch_separation_ratio
                    ),
                    "tactile_patch_thickness_ratio": float(tactile_patch_thickness_ratio),
                    "patch_min_normal_cos": float(patch_min_normal_cos),
                    "tactile_patch_dominant_normal_gap_cos": float(tactile_patch_dominant_normal_gap_cos),
                    "tactile_patch_plane_gap_ratio": float(tactile_patch_plane_gap_ratio),
                    "tactile_patch_link_radius_ratio": float(tactile_patch_link_radius_ratio),
                    "max_target_contact_offset_ratio": float(max_target_contact_offset_ratio),
                    "tactile_reachable_clearance_ratio": float(tactile_reachable_clearance_ratio),
                    "tactile_reachable_approach_steps": int(tactile_reachable_approach_steps),
                    "normalization_bound": float(normalization_bound),
                    "num_surface_points": int(num_surface_points),
                    "num_query_points": int(num_query_points),
                    "query_uniform_region": str(query_uniform_region),
                    "query_sampling_mode": str(query_sampling_mode),
                    "paired_query_fraction": float(paired_query_fraction),
                    "paired_query_eps_min": float(paired_query_eps_min),
                    "paired_query_eps_max": float(paired_query_eps_max),
                    "paired_query_max_attempts": int(paired_query_max_attempts),
                    "paired_query_anchor_mode": str(paired_query_anchor_mode),
                    "paired_query_coverage_grid_size": int(paired_query_coverage_grid_size),
                    "paired_query_coverage_min_per_cell": int(paired_query_coverage_min_per_cell),
                    "paired_query_eps_retries": int(paired_query_eps_retries),
                    "watertight_proxy_mode": str(watertight_proxy_mode),
                    "watertight_mesh_usage": str(watertight_mesh_usage),
                    "non_watertight_policy": str(non_watertight_policy),
                    "proxy_poisson_samples": int(proxy_poisson_samples),
                    "proxy_poisson_depth": int(proxy_poisson_depth),
                    "proxy_poisson_full_depth": int(proxy_poisson_full_depth),
                    "proxy_poisson_threads": int(proxy_poisson_threads),
                    "manifoldplus_path": None if manifoldplus_path is None else str(manifoldplus_path),
                    "manifoldplus_depth": int(manifoldplus_depth),
                    "mujoco_max_faces": int(mujoco_max_faces),
                    "query_occupancy_nsamples": int(query_occupancy_nsamples),
                    "query_near_surface_sign_band": float(query_near_surface_sign_band),
                    "seed": int(object_seed(base_seed, obj_path)),
                }
            )
    return category_dirs, jobs


def run_parallel_jobs(jobs, max_workers=1, fail_fast=False):
    if not jobs:
        return []

    max_workers = max(1, int(max_workers))
    if max_workers == 1:
        return [process_single_obj_job(job) for job in jobs]

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(process_single_obj_job, job): job for job in jobs
        }
        for future in concurrent.futures.as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "status": "failed",
                    "obj_path": str(job["obj_path"]),
                    "out_path": str(job["out_path"]),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            results.append(result)
            if result["status"] == "failed" and bool(fail_fast):
                for pending in future_to_job:
                    pending.cancel()
                break
    return results


def summarise_results(results):
    summary = {
        "ok": 0,
        "skipped": 0,
        "failed": 0,
        "mean_coverage": 0.0,
        "mean_reachable_surface_fraction": 0.0,
        "query_proxy_watertight": 0,
        "query_proxy_non_watertight": 0,
        "mean_query_inside_ratio": 0.0,
        "outputs": [],
        "failures": [],
    }
    coverages = []
    reachable_fractions = []
    query_inside_ratios = []
    for result in results:
        status = result["status"]
        summary[status] += 1
        if status == "ok":
            summary["outputs"].append(result["out_path"])
            coverages.append(float(result.get("coverage", 0.0)))
            reachable_fractions.append(float(result.get("reachable_fraction", 0.0)))
            if bool(result.get("query_proxy_is_watertight", False)):
                summary["query_proxy_watertight"] += 1
            else:
                summary["query_proxy_non_watertight"] += 1
            inside_count = int(result.get("query_inside_count", 0))
            outside_count = int(result.get("query_outside_count", 0))
            total_query_count = max(1, inside_count + outside_count)
            query_inside_ratios.append(float(inside_count) / float(total_query_count))
        elif status == "failed":
            summary["failures"].append(
                {
                    "obj_path": result["obj_path"],
                    "error": result.get("error"),
                    "traceback": result.get("traceback"),
                }
            )
    if coverages:
        summary["mean_coverage"] = float(np.mean(coverages))
    if reachable_fractions:
        summary["mean_reachable_surface_fraction"] = float(np.mean(reachable_fractions))
    if query_inside_ratios:
        summary["mean_query_inside_ratio"] = float(np.mean(query_inside_ratios))
    return summary


def process_shapenetcore_all_categories_mujoco_coverage_onefolder(
    root_dir,
    category_names=None,
    max_objects_per_category=None,
    num_tactile_samples=10,
    tactile_num_fingers=10,
    tactile_points_per_finger=3000,
    dense_surface_sample_n=120000,
    candidate_touch_samples=6000,
    tactile_patch_radius_ratio=0.10,
    tactile_min_touch_separation_ratio=0.055,
    tactile_patch_thickness_ratio=0.035,
    patch_min_normal_cos=0.05,
    tactile_patch_dominant_normal_gap_cos=0.18,
    tactile_patch_plane_gap_ratio=0.35,
    tactile_patch_link_radius_ratio=0.0,
    max_target_contact_offset_ratio=0.60,
    tactile_reachable_clearance_ratio=0.92,
    tactile_reachable_approach_steps=5,
    normalization_bound=0.9,
    num_surface_points=235000,
    num_query_points=250000,
    query_uniform_region="cube",
    query_sampling_mode="paired_normal_offsets",
    paired_query_fraction=0.90,
    paired_query_eps_min=0.002,
    paired_query_eps_max=0.025,
    paired_query_max_attempts=8,
    paired_query_anchor_mode="coverage_grid",
    paired_query_coverage_grid_size=12,
    paired_query_coverage_min_per_cell=1,
    paired_query_eps_retries=3,
    watertight_proxy_mode="repair",
    watertight_mesh_usage="sign_proxy",
    non_watertight_policy="skip",
    proxy_poisson_samples=50000,
    proxy_poisson_depth=8,
    proxy_poisson_full_depth=5,
    proxy_poisson_threads=8,
    manifoldplus_path=None,
    manifoldplus_depth=8,
    mujoco_max_faces=190000,
    query_occupancy_nsamples=11,
    query_near_surface_sign_band=0.01,
    output_folder_name="tactistruct_npz_shapenet_mujoco_coverage_onefolder_paired_watertight_strict",
    asset_folder_name="tactistruct_npz_shapenet_mujoco_coverage_assets_paired_watertight_strict",
    max_workers=1,
    overwrite=False,
    fail_fast=False,
    base_seed=42,
):
    category_dirs, jobs = build_jobs(
        root_dir=root_dir,
        category_names=category_names,
        max_objects_per_category=max_objects_per_category,
        num_tactile_samples=num_tactile_samples,
        tactile_num_fingers=tactile_num_fingers,
        tactile_points_per_finger=tactile_points_per_finger,
        dense_surface_sample_n=dense_surface_sample_n,
        candidate_touch_samples=candidate_touch_samples,
        tactile_patch_radius_ratio=tactile_patch_radius_ratio,
        tactile_min_touch_separation_ratio=tactile_min_touch_separation_ratio,
        tactile_patch_thickness_ratio=tactile_patch_thickness_ratio,
        patch_min_normal_cos=patch_min_normal_cos,
        tactile_patch_dominant_normal_gap_cos=tactile_patch_dominant_normal_gap_cos,
        tactile_patch_plane_gap_ratio=tactile_patch_plane_gap_ratio,
        tactile_patch_link_radius_ratio=tactile_patch_link_radius_ratio,
        max_target_contact_offset_ratio=max_target_contact_offset_ratio,
        tactile_reachable_clearance_ratio=tactile_reachable_clearance_ratio,
        tactile_reachable_approach_steps=tactile_reachable_approach_steps,
        normalization_bound=normalization_bound,
        num_surface_points=num_surface_points,
        num_query_points=num_query_points,
        query_uniform_region=query_uniform_region,
        query_sampling_mode=query_sampling_mode,
        paired_query_fraction=paired_query_fraction,
        paired_query_eps_min=paired_query_eps_min,
        paired_query_eps_max=paired_query_eps_max,
        paired_query_max_attempts=paired_query_max_attempts,
        paired_query_anchor_mode=paired_query_anchor_mode,
        paired_query_coverage_grid_size=paired_query_coverage_grid_size,
        paired_query_coverage_min_per_cell=paired_query_coverage_min_per_cell,
        paired_query_eps_retries=paired_query_eps_retries,
        watertight_proxy_mode=watertight_proxy_mode,
        watertight_mesh_usage=watertight_mesh_usage,
        non_watertight_policy=non_watertight_policy,
        proxy_poisson_samples=proxy_poisson_samples,
        proxy_poisson_depth=proxy_poisson_depth,
        proxy_poisson_full_depth=proxy_poisson_full_depth,
        proxy_poisson_threads=proxy_poisson_threads,
        manifoldplus_path=manifoldplus_path,
        manifoldplus_depth=manifoldplus_depth,
        mujoco_max_faces=mujoco_max_faces,
        query_occupancy_nsamples=query_occupancy_nsamples,
        query_near_surface_sign_band=query_near_surface_sign_band,
        output_folder_name=output_folder_name,
        asset_folder_name=asset_folder_name,
        overwrite=overwrite,
        base_seed=base_seed,
    )

    if not category_dirs:
        print("[WARN] no category folders found under:", root_dir)
        return {"ok": 0, "skipped": 0, "failed": 0, "mean_coverage": 0.0}

    out_dir = os.path.join(root_dir, output_folder_name)
    asset_dir = os.path.join(root_dir, asset_folder_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(asset_dir, exist_ok=True)

    print(f"[INFO] found {len(category_dirs)} category folders under root.")
    print(f"[INFO] flat output dir : {out_dir}")
    print(f"[INFO] MuJoCo asset dir: {asset_dir}")
    print(f"[INFO] prepared {len(jobs)} object jobs")
    print(f"[INFO] max_workers={int(max_workers)} overwrite={bool(overwrite)}")

    results = run_parallel_jobs(
        jobs=jobs,
        max_workers=max_workers,
        fail_fast=fail_fast,
    )
    for result in results:
        status = result["status"].upper()
        if result["status"] == "ok":
            print(
                f"[{status}] {result['obj_path']} -> {result['out_path']} "
                f"(coverage={result['coverage']:.4f}, "
                f"reachable_fraction={result['reachable_fraction']:.4f}, "
                f"mesh={result['mesh_geometry_source']}, "
                f"sign_proxy={result['query_proxy_source']}, "
                f"watertight={result['query_proxy_is_watertight']})"
            )
        elif result["status"] == "skipped":
            print(f"[{status}] {result['obj_path']} ({result['message']})")
        else:
            print(f"[{status}] {result['obj_path']}")
            print(result.get("error", "unknown error"))

    summary = summarise_results(results)
    summary_path = os.path.join(out_dir, "preprocess_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        import json

        json.dump(summary, handle, indent=2)
    print(f"[SUMMARY] saved to {summary_path}")
    print(
        f"[SUMMARY] ok={summary['ok']} skipped={summary['skipped']} "
        f"failed={summary['failed']} mean_coverage={summary['mean_coverage']:.4f} "
        f"mean_reachable_surface_fraction={summary['mean_reachable_surface_fraction']:.4f} "
        f"proxy_watertight={summary['query_proxy_watertight']} "
        f"mean_inside_ratio={summary['mean_query_inside_ratio']:.4f}"
    )
    if summary["failed"] > 0 and bool(fail_fast):
        raise RuntimeError("At least one preprocessing job failed in fail-fast mode.")
    return summary


if __name__ == "__main__":
    multiprocessing.freeze_support()
    args = parse_args()
    process_shapenetcore_all_categories_mujoco_coverage_onefolder(
        root_dir=str(Path(args.root_dir).resolve()),
        category_names=parse_category_names(args.category_names),
        max_objects_per_category=args.max_objects_per_category,
        num_tactile_samples=args.num_tactile_samples,
        tactile_num_fingers=args.tactile_num_fingers,
        tactile_points_per_finger=args.tactile_points_per_finger,
        dense_surface_sample_n=args.dense_surface_sample_n,
        candidate_touch_samples=args.candidate_touch_samples,
        tactile_patch_radius_ratio=args.tactile_patch_radius_ratio,
        tactile_min_touch_separation_ratio=args.tactile_min_touch_separation_ratio,
        tactile_patch_thickness_ratio=args.tactile_patch_thickness_ratio,
        patch_min_normal_cos=args.patch_min_normal_cos,
        tactile_patch_dominant_normal_gap_cos=args.tactile_patch_dominant_normal_gap_cos,
        tactile_patch_plane_gap_ratio=args.tactile_patch_plane_gap_ratio,
        tactile_patch_link_radius_ratio=args.tactile_patch_link_radius_ratio,
        max_target_contact_offset_ratio=args.max_target_contact_offset_ratio,
        tactile_reachable_clearance_ratio=args.tactile_reachable_clearance_ratio,
        tactile_reachable_approach_steps=args.tactile_reachable_approach_steps,
        normalization_bound=args.normalization_bound,
        num_surface_points=args.num_surface_points,
        num_query_points=args.num_query_points,
        query_uniform_region=str(args.query_uniform_region),
        query_sampling_mode=str(args.query_sampling_mode),
        paired_query_fraction=args.paired_query_fraction,
        paired_query_eps_min=args.paired_query_eps_min,
        paired_query_eps_max=args.paired_query_eps_max,
        paired_query_max_attempts=args.paired_query_max_attempts,
        paired_query_anchor_mode=str(args.paired_query_anchor_mode),
        paired_query_coverage_grid_size=args.paired_query_coverage_grid_size,
        paired_query_coverage_min_per_cell=args.paired_query_coverage_min_per_cell,
        paired_query_eps_retries=args.paired_query_eps_retries,
        watertight_proxy_mode=str(args.watertight_proxy_mode),
        watertight_mesh_usage=str(args.watertight_mesh_usage),
        non_watertight_policy=str(args.non_watertight_policy),
        proxy_poisson_samples=args.proxy_poisson_samples,
        proxy_poisson_depth=args.proxy_poisson_depth,
        proxy_poisson_full_depth=args.proxy_poisson_full_depth,
        proxy_poisson_threads=args.proxy_poisson_threads,
        manifoldplus_path=args.manifoldplus_path,
        manifoldplus_depth=args.manifoldplus_depth,
        mujoco_max_faces=args.mujoco_max_faces,
        query_occupancy_nsamples=args.query_occupancy_nsamples,
        query_near_surface_sign_band=args.query_near_surface_sign_band,
        output_folder_name=args.output_folder_name,
        asset_folder_name=args.asset_folder_name,
        max_workers=args.max_workers,
        overwrite=bool(args.overwrite),
        fail_fast=bool(args.fail_fast),
        base_seed=args.base_seed,
    )
    print("\nAll done.")
