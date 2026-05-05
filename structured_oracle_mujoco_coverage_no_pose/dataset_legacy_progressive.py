from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from tactistruct.data.dataset import _sample_indices, _sample_query_indices, _subsample_rows


def flatten_structured_view_touch_points(payload, view_index: int) -> np.ndarray:
    patch_points_view = np.asarray(payload["patch_points"], dtype=np.float32)[view_index]
    patch_mask_view = np.asarray(payload["patch_mask"], dtype=bool)[view_index]
    patch_counts_view = np.asarray(payload["patch_counts"], dtype=np.int32)[view_index]

    if patch_points_view.ndim != 4:
        raise ValueError(
            f"Expected patch_points[view] with shape [F, R, P, 3], got {patch_points_view.shape}."
        )

    num_fingers, num_rounds, points_per_patch, _ = patch_points_view.shape
    chunks = []
    for finger_index in range(num_fingers):
        for round_index in range(num_rounds):
            if not bool(patch_mask_view[finger_index, round_index]):
                continue
            stored_patch = patch_points_view[finger_index, round_index]
            source_count = int(patch_counts_view[finger_index, round_index])
            kept_count = min(points_per_patch, max(1, source_count))
            chunks.append(stored_patch[:kept_count].astype(np.float32))

    if not chunks:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32)


class StructuredLegacyProgressiveDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = ".",
        object_filenames: list[str] | None = None,
        recursive: bool = True,
        touch_view_indices: list[int] | None = None,
        num_surface_points: int = 4096,
        num_touch_points: int = 512,
        num_query_points: int = 6144,
        dynamic_sampling: bool = True,
        near_surface_ratio: float = 0.6,
        near_surface_threshold: float = 0.015,
        negative_sample_ratio: float = 0.6,
        seed: int = 42,
        cache_in_memory: bool = False,
        min_touch_points: int | None = None,
        merge_touch_views_prob: float = 0.0,
        max_merged_views: int = 1,
    ) -> None:
        super().__init__()
        self.root = Path(root).resolve()
        self.base_dir = (self.root / split).resolve()
        self.recursive = bool(recursive)
        self.num_surface_points = int(num_surface_points)
        self.num_touch_points = int(num_touch_points)
        self.num_query_points = int(num_query_points)
        self.dynamic_sampling = bool(dynamic_sampling)
        self.near_surface_ratio = float(near_surface_ratio)
        self.near_surface_threshold = float(near_surface_threshold)
        self.negative_sample_ratio = float(negative_sample_ratio)
        self.seed = int(seed)
        self.cache_in_memory = bool(cache_in_memory)
        self.touch_view_indices = None if touch_view_indices is None else sorted({int(i) for i in touch_view_indices})
        if min_touch_points is None:
            min_touch_points = self.num_touch_points
        self.min_touch_points = max(1, min(int(min_touch_points), self.num_touch_points))
        self.merge_touch_views_prob = float(np.clip(merge_touch_views_prob, 0.0, 1.0))
        self.max_merged_views = max(1, int(max_merged_views))
        self.cached_bytes = 0
        self.memory_cache: list[dict[str, np.ndarray]] | None = None

        all_files = sorted(self.base_dir.rglob("*.npz") if self.recursive else self.base_dir.glob("*.npz"))
        if object_filenames:
            relative_map = {path.relative_to(self.base_dir).as_posix(): path for path in all_files}
            basename_map: dict[str, list[Path]] = {}
            for path in all_files:
                basename_map.setdefault(path.name, []).append(path)
            selected_files = []
            for name in object_filenames:
                normalized = Path(name).as_posix().lstrip("./")
                if normalized in relative_map:
                    selected_files.append(relative_map[normalized])
                    continue
                matches = basename_map.get(Path(normalized).name, [])
                if len(matches) == 1:
                    selected_files.append(matches[0])
            if selected_files:
                all_files = selected_files

        if not all_files:
            raise FileNotFoundError(f"No .npz files found in {self.base_dir}")

        self.files = all_files
        self.file_view_specs: list[list[int]] = []
        self.index_map: list[tuple[int, int]] = []
        for file_index, file_path in enumerate(self.files):
            with np.load(file_path, mmap_mode="r") as payload:
                if "patch_points" not in payload:
                    raise KeyError(f"{file_path} does not contain 'patch_points'.")
                view_count = int(payload["patch_points"].shape[0])
                selected_views = list(range(view_count)) if self.touch_view_indices is None else list(self.touch_view_indices)
                invalid = [index for index in selected_views if index < 0 or index >= view_count]
                if invalid:
                    raise IndexError(f"{file_path.name} only has {view_count} touch views, invalid indices: {invalid}")
                self.file_view_specs.append(selected_views)
                self.index_map.extend((file_index, view_index) for view_index in selected_views)

        if self.cache_in_memory:
            self.memory_cache = [self._load_payload_into_memory(file_path) for file_path in self.files]
            self.cached_bytes = int(
                sum(
                    array.nbytes
                    for payload in self.memory_cache
                    for array in payload.values()
                    if isinstance(array, np.ndarray)
                )
            )

    def __len__(self) -> int:
        return len(self.index_map)

    def _make_rng(self, index: int) -> np.random.Generator:
        if self.dynamic_sampling:
            return np.random.default_rng()
        return np.random.default_rng(self.seed + index)

    def _cache_keys_for_payload(self, payload) -> list[str]:
        keys = {
            "surface_points",
            "query_points",
            "query_sdf",
            "patch_points",
            "patch_mask",
            "patch_counts",
        }
        if "surface_normals" in payload:
            keys.add("surface_normals")
        return sorted(keys)

    def _load_payload_into_memory(self, file_path: Path) -> dict[str, np.ndarray]:
        with np.load(file_path) as payload:
            return {
                key: np.array(payload[key], copy=True)
                for key in self._cache_keys_for_payload(payload)
                if key in payload
            }

    def _sample_surface_and_queries(
        self,
        payload,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        full_surface_points = np.asarray(payload["surface_points"], dtype=np.float32)
        surface_indices = _sample_indices(len(full_surface_points), self.num_surface_points, rng)
        surface_points = full_surface_points[surface_indices]

        query_sdf_full = np.asarray(payload["query_sdf"], dtype=np.float32)
        query_indices = _sample_query_indices(
            query_sdf_full,
            self.num_query_points,
            rng,
            near_surface_ratio=self.near_surface_ratio,
            near_surface_threshold=self.near_surface_threshold,
            negative_sample_ratio=self.negative_sample_ratio,
        )
        query_points = np.asarray(payload["query_points"], dtype=np.float32)[query_indices]
        query_sdf = query_sdf_full[query_indices]
        return surface_points, surface_indices, query_points, query_sdf

    def _select_touch_views(self, file_index: int, selected_view_index: int, rng: np.random.Generator) -> list[int]:
        selected_views = [selected_view_index]
        available_views = [view for view in self.file_view_specs[file_index] if view != selected_view_index]
        if self.merge_touch_views_prob <= 0.0 or self.max_merged_views <= 1:
            return selected_views
        if not available_views or rng.random() >= self.merge_touch_views_prob:
            return selected_views
        max_views = min(self.max_merged_views, len(available_views) + 1)
        target_view_count = int(rng.integers(2, max_views + 1))
        extra_count = target_view_count - 1
        extra_indices = rng.choice(len(available_views), size=extra_count, replace=False)
        selected_views.extend(available_views[int(index)] for index in np.atleast_1d(extra_indices).tolist())
        return selected_views

    def _sample_touch_points(
        self,
        payload,
        selected_views: list[int],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int, int]:
        raw_chunks = []
        for view_index in selected_views:
            raw_points = flatten_structured_view_touch_points(payload, view_index)
            if len(raw_points) > 0:
                raw_chunks.append(raw_points)
        if not raw_chunks:
            raise ValueError("Could not load any tactile points for the selected structured views.")

        merged_touch_points = np.concatenate(raw_chunks, axis=0).astype(np.float32)
        if self.min_touch_points >= self.num_touch_points:
            effective_count = int(self.num_touch_points)
        else:
            effective_count = int(rng.integers(self.min_touch_points, self.num_touch_points + 1))

        effective_touch_points = _subsample_rows(merged_touch_points, effective_count, rng)
        fixed_touch_points = _subsample_rows(effective_touch_points, self.num_touch_points, rng).astype(np.float32)
        return fixed_touch_points, effective_count, len(selected_views)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_index, view_index = self.index_map[index]
        rng = self._make_rng(index)

        payload = self.memory_cache[file_index] if self.memory_cache is not None else None
        if payload is None:
            with np.load(self.files[file_index]) as loaded:
                payload = {
                    key: np.asarray(loaded[key])
                    for key in self._cache_keys_for_payload(loaded)
                    if key in loaded
                }

        surface_points, surface_indices, query_points, query_sdf = self._sample_surface_and_queries(payload, rng)
        selected_views = self._select_touch_views(file_index, view_index, rng)
        touch_points, effective_count, merged_view_count = self._sample_touch_points(payload, selected_views, rng)

        sample: dict[str, torch.Tensor] = {
            "surface_points": torch.from_numpy(surface_points.astype(np.float32)),
            "query_points": torch.from_numpy(query_points.astype(np.float32)),
            "query_sdf": torch.from_numpy(query_sdf.astype(np.float32)),
            "touch_points": torch.from_numpy(touch_points.astype(np.float32)),
            "touch_effective_count": torch.tensor(effective_count, dtype=torch.long),
            "touch_count_ratio": torch.tensor(float(effective_count) / float(max(self.num_touch_points, 1)), dtype=torch.float32),
            "touch_merge_count": torch.tensor(merged_view_count, dtype=torch.long),
            "object_index": torch.tensor(file_index, dtype=torch.long),
            "touch_view_index": torch.tensor(view_index, dtype=torch.long),
        }
        if "surface_normals" in payload:
            sample["surface_normals"] = torch.from_numpy(
                np.asarray(payload["surface_normals"], dtype=np.float32)[surface_indices]
            )
        return sample


def load_structured_touch_points_from_file(
    touch_file: str | Path,
    touch_view_indices: list[int] | None = None,
) -> tuple[torch.Tensor, dict[str, int | str | None]]:
    touch_file = Path(touch_file)
    with np.load(touch_file) as payload:
        if "patch_points" not in payload:
            raise KeyError(f"Could not find 'patch_points' in {touch_file}.")
        available_count = int(payload["patch_points"].shape[0])
        selected_indices = list(range(available_count)) if touch_view_indices is None else [int(index) for index in touch_view_indices]
        invalid_indices = [index for index in selected_indices if index < 0 or index >= available_count]
        if invalid_indices:
            raise IndexError(
                f"touch_view_indices contains out-of-range values {invalid_indices} "
                f"for {touch_file}, which has {available_count} tactile views."
            )
        merged_touch_points = [flatten_structured_view_touch_points(payload, view_index) for view_index in selected_indices]
        merged_touch_points = np.concatenate(merged_touch_points, axis=0).astype(np.float32)
        metadata = {
            "touch_key": "patch_points_flattened",
            "touch_view_index": None if len(selected_indices) != 1 else int(selected_indices[0]),
            "touch_group_key": None,
            "touch_group_value": None,
            "touch_label": "merged" if len(selected_indices) > 1 else f"view{int(selected_indices[0]):02d}",
            "num_touch_points": int(merged_touch_points.shape[0]),
        }
        return torch.from_numpy(merged_touch_points).unsqueeze(0), metadata


__all__ = [
    "StructuredLegacyProgressiveDataset",
    "flatten_structured_view_touch_points",
    "load_structured_touch_points_from_file",
]
