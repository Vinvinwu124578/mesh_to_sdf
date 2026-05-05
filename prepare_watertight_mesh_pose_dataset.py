from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import trimesh


MESH_EXTENSIONS = {".stl", ".obj", ".ply", ".off"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a pose-augmented mesh point dataset from watertight meshes. "
            "Each output NPZ stores posed surface points, normals, and pose metadata."
        )
    )
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--poses-per-object", type=int, default=32)
    parser.add_argument("--surface-points", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--full-rotation", action="store_true")
    parser.add_argument("--max-degrees", type=float, default=180.0)
    parser.add_argument("--include-identity", action="store_true")
    parser.add_argument(
        "--rotate-around",
        type=str,
        default="bbox_center",
        choices=("bbox_center", "centroid", "origin"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--export-posed-meshes",
        action="store_true",
        help="Also export one posed STL per pose. This can take a lot of disk space.",
    )
    return parser.parse_args()


def collect_mesh_files(input_dir: Path) -> list[Path]:
    files = [
        path
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in MESH_EXTENSIONS
    ]
    if not files:
        raise FileNotFoundError(f"No mesh files found under {input_dir}")
    return files


def load_stl_mesh(path: Path) -> trimesh.Trimesh:
    raw = path.read_bytes()
    if len(raw) >= 84:
        face_count = int(np.frombuffer(raw[80:84], dtype="<u4", count=1)[0])
        expected_size = 84 + face_count * 50
        if expected_size == len(raw):
            dtype = np.dtype(
                [
                    ("normal", "<f4", (3,)),
                    ("vertices", "<f4", (3, 3)),
                    ("attr", "<u2"),
                ]
            )
            records = np.frombuffer(raw, dtype=dtype, count=face_count, offset=84)
            vertices = np.asarray(records["vertices"], dtype=np.float32).reshape(-1, 3).copy()
            faces = np.arange(face_count * 3, dtype=np.int64).reshape(-1, 3)
            normals = np.asarray(records["normal"], dtype=np.float32).copy()
            return trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                face_normals=normals,
                process=False,
            )

    vertices = []
    current_vertices = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].lower() == "vertex":
                current_vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(current_vertices) == 3:
                    vertices.extend(current_vertices)
                    current_vertices = []

    if not vertices:
        raise ValueError(f"Could not parse STL mesh from {path}")
    face_count = len(vertices) // 3
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.arange(face_count * 3, dtype=np.int64).reshape(-1, 3),
        process=False,
    )


def load_mesh(path: Path) -> trimesh.Trimesh:
    if path.suffix.lower() == ".stl":
        return load_stl_mesh(path)

    loaded = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geom
            for geom in loaded.geometry.values()
            if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0
        ]
        if not meshes:
            raise ValueError(f"No triangle mesh geometry found in {path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"Could not load a valid triangle mesh from {path}")
    return loaded


def check_watertight_after_merge(mesh: trimesh.Trimesh) -> bool:
    try:
        merged = mesh.copy()
        merged.merge_vertices(digits_vertex=6)
        return bool(merged.is_watertight)
    except Exception:
        return bool(mesh.is_watertight)


def _uniform_random_rotation(rng: np.random.Generator) -> np.ndarray:
    quat_xyzw = rng.normal(size=4).astype(np.float32)
    norm = np.linalg.norm(quat_xyzw)
    if norm <= 1e-8:
        quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    else:
        quat_xyzw = quat_xyzw / norm
    x, y, z, w = quat_xyzw
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _axis_angle_rotation(rng: np.random.Generator, max_degrees: float) -> np.ndarray:
    axis = rng.normal(size=3).astype(np.float32)
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 1e-8:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        axis = axis / axis_norm
    angle = float(rng.uniform(-math.radians(max_degrees), math.radians(max_degrees)))
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one_minus_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float32,
    )


def sample_rotation(rng: np.random.Generator, full_rotation: bool, max_degrees: float) -> np.ndarray:
    if full_rotation or max_degrees >= 179.999:
        return _uniform_random_rotation(rng)
    return _axis_angle_rotation(rng, max_degrees=max_degrees)


def rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    quat = np.asarray([w, x, y, z], dtype=np.float32)
    return quat / np.clip(np.linalg.norm(quat), 1e-6, None)


def mesh_bounds(mesh: trimesh.Trimesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3)
    return np.stack([np.min(vertices, axis=0), np.max(vertices, axis=0)], axis=0)


def rotation_center(mesh: trimesh.Trimesh, mode: str) -> np.ndarray:
    mode = str(mode).strip().lower()
    bounds = mesh_bounds(mesh)
    if mode == "origin":
        return np.zeros((3,), dtype=np.float32)
    if mode == "centroid":
        return np.mean(np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3), axis=0)
    return np.mean(bounds, axis=0).astype(np.float32)


def apply_pose(points: np.ndarray, rotation: np.ndarray, center: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32).reshape(3)
    flat = points.reshape(-1, 3)
    local = flat - center.reshape(1, 3)
    rotated = np.empty_like(local, dtype=np.float32)
    rotated[:, 0] = (
        local[:, 0] * rotation[0, 0]
        + local[:, 1] * rotation[0, 1]
        + local[:, 2] * rotation[0, 2]
        + center[0]
    )
    rotated[:, 1] = (
        local[:, 0] * rotation[1, 0]
        + local[:, 1] * rotation[1, 1]
        + local[:, 2] * rotation[1, 2]
        + center[1]
    )
    rotated[:, 2] = (
        local[:, 0] * rotation[2, 0]
        + local[:, 1] * rotation[2, 1]
        + local[:, 2] * rotation[2, 2]
        + center[2]
    )
    return rotated.reshape(points.shape).astype(np.float32)


def rotate_normals(normals: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    flat = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
    rotated = np.empty_like(flat, dtype=np.float32)
    rotated[:, 0] = (
        flat[:, 0] * rotation[0, 0]
        + flat[:, 1] * rotation[0, 1]
        + flat[:, 2] * rotation[0, 2]
    )
    rotated[:, 1] = (
        flat[:, 0] * rotation[1, 0]
        + flat[:, 1] * rotation[1, 1]
        + flat[:, 2] * rotation[1, 2]
    )
    rotated[:, 2] = (
        flat[:, 0] * rotation[2, 0]
        + flat[:, 1] * rotation[2, 1]
        + flat[:, 2] * rotation[2, 2]
    )
    norms = np.linalg.norm(rotated, axis=1, keepdims=True)
    rotated = rotated / np.clip(norms, 1e-8, None)
    return rotated.reshape(normals.shape).astype(np.float32)


def rotate_vector(vector: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float32).reshape(3)
    return np.asarray(
        [
            x * rotation[0, 0] + y * rotation[0, 1] + z * rotation[0, 2],
            x * rotation[1, 0] + y * rotation[1, 1] + z * rotation[1, 2],
            x * rotation[2, 0] + y * rotation[2, 1] + z * rotation[2, 2],
        ],
        dtype=np.float32,
    )


def sample_surface_payload(mesh: trimesh.Trimesh, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    state = np.random.get_state()
    np.random.seed(int(seed) % (2**32 - 1))
    try:
        points, face_ids = trimesh.sample.sample_surface(mesh, int(count))
    finally:
        np.random.set_state(state)
    normals = np.asarray(mesh.face_normals[face_ids], dtype=np.float32)
    return points.astype(np.float32), normals.astype(np.float32)


def safe_name(path: Path) -> str:
    return path.stem.replace(" ", "_").replace("__normalized", "")


def export_posed_mesh(mesh: trimesh.Trimesh, rotation: np.ndarray, center: np.ndarray, output_path: Path) -> None:
    posed_mesh = mesh.copy()
    posed_mesh.vertices = apply_pose(np.asarray(posed_mesh.vertices, dtype=np.float32), rotation, center)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    posed_mesh.export(output_path)


def process_mesh_file(
    mesh_path: Path,
    output_dir: Path,
    object_index: int,
    args: argparse.Namespace,
) -> int:
    mesh = load_mesh(mesh_path)
    mesh_is_watertight = check_watertight_after_merge(mesh)
    object_name = safe_name(mesh_path)
    object_dir = output_dir / object_name
    mesh_dir = object_dir / "posed_meshes"
    npz_dir = object_dir / "npz"
    center = rotation_center(mesh, args.rotate_around)
    bbox = mesh_bounds(mesh).astype(np.float32)
    surface_points, surface_normals = sample_surface_payload(
        mesh,
        count=int(args.surface_points),
        seed=int(args.seed) + 1009 * int(object_index),
    )

    converted = 0
    poses_per_object = max(1, int(args.poses_per_object))
    for pose_index in range(poses_per_object):
        if pose_index == 0 and bool(args.include_identity):
            rotation = np.eye(3, dtype=np.float32)
        else:
            pose_seed = int(args.seed) + 1009 * int(object_index) + 37 * int(pose_index)
            rng = np.random.default_rng(pose_seed)
            rotation = sample_rotation(
                rng,
                full_rotation=bool(args.full_rotation),
                max_degrees=float(args.max_degrees),
            )

        output_npz = npz_dir / f"{object_name}__pose_{pose_index:03d}.npz"
        output_mesh = mesh_dir / f"{object_name}__pose_{pose_index:03d}.stl"
        if output_npz.exists() and not bool(args.overwrite):
            continue

        posed_points = apply_pose(surface_points, rotation, center)
        posed_normals = rotate_normals(surface_normals, rotation)
        pose_translation = center - rotate_vector(center, rotation)
        pose_transform = np.eye(4, dtype=np.float32)
        pose_transform[:3, :3] = rotation.astype(np.float32)
        pose_transform[:3, 3] = pose_translation.astype(np.float32)

        npz_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_npz,
            surface_points=posed_points.astype(np.float32),
            surface_normals=posed_normals.astype(np.float32),
            pose_rotation=rotation.astype(np.float32),
            pose_quaternion_wxyz=rotation_matrix_to_quaternion_wxyz(rotation),
            pose_transform=pose_transform,
            pose_translation=pose_translation.astype(np.float32),
            pose_center=center.astype(np.float32),
            pose_index=np.asarray(pose_index, dtype=np.int32),
            object_index=np.asarray(object_index, dtype=np.int32),
            object_name=np.asarray(object_name),
            source_mesh=np.asarray(str(mesh_path)),
            posed_mesh=np.asarray(str(output_mesh) if args.export_posed_meshes else ""),
            mesh_is_watertight=np.asarray(bool(mesh_is_watertight)),
            mesh_vertex_count=np.asarray(int(len(mesh.vertices)), dtype=np.int32),
            mesh_face_count=np.asarray(int(len(mesh.faces)), dtype=np.int32),
            mesh_bounds=bbox,
            rotate_around=np.asarray(str(args.rotate_around)),
        )

        if bool(args.export_posed_meshes):
            export_posed_mesh(mesh, rotation, center, output_mesh)

        converted += 1
        print(f"[pose-mesh] saved {output_npz}")

    return converted


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    mesh_files = collect_mesh_files(input_dir)
    if int(args.max_files) > 0:
        mesh_files = mesh_files[: int(args.max_files)]

    total_converted = 0
    object_summaries = []
    for object_index, mesh_path in enumerate(mesh_files):
        converted = process_mesh_file(mesh_path, output_dir, object_index, args)
        total_converted += int(converted)
        object_summaries.append(
            {
                "object_index": int(object_index),
                "mesh_path": str(mesh_path),
                "converted_poses": int(converted),
            }
        )

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_objects": int(len(mesh_files)),
        "poses_per_object": int(args.poses_per_object),
        "surface_points": int(args.surface_points),
        "converted_files": int(total_converted),
        "full_rotation": bool(args.full_rotation),
        "max_degrees": float(args.max_degrees),
        "include_identity": bool(args.include_identity),
        "rotate_around": str(args.rotate_around),
        "export_posed_meshes": bool(args.export_posed_meshes),
        "objects": object_summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "mesh_pose_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[pose-mesh] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
