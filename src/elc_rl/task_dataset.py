"""Typed loader for measured FRF artifacts used by offline diagnostics only."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_TASK_ID = "cgs_turntable_001"


@dataclass(frozen=True)
class FRFTask:
    task_id: str
    current_frf: np.ndarray
    speed_frf: np.ndarray
    position_frf: np.ndarray
    speed_amplitudes_mA: np.ndarray
    speed_roles: np.ndarray
    speed_suspicious_mask: np.ndarray
    base_feature_names: tuple[str, ...]
    speed_feature_names: tuple[str, ...]
    context_vector: np.ndarray
    manifest: dict[str, Any]

    @property
    def total_points(self) -> int:
        return int(
            self.current_frf.shape[0]
            + self.speed_frf.shape[0] * self.speed_frf.shape[1]
            + self.position_frf.shape[0]
        )

    def validate(self) -> None:
        if self.task_id != EXPECTED_TASK_ID:
            raise ValueError(f"unexpected task_id: {self.task_id}")
        if self.current_frf.shape != (18, 5):
            raise ValueError(f"unexpected current_frf shape: {self.current_frf.shape}")
        if self.speed_frf.shape != (6, 15, 6):
            raise ValueError(f"unexpected speed_frf shape: {self.speed_frf.shape}")
        if self.position_frf.shape != (15, 5):
            raise ValueError(f"unexpected position_frf shape: {self.position_frf.shape}")
        if self.speed_suspicious_mask.shape != (6, 15):
            raise ValueError(
                f"unexpected speed_suspicious_mask shape: {self.speed_suspicious_mask.shape}"
            )
        if self.context_vector.shape != (705,):
            raise ValueError(f"unexpected context shape: {self.context_vector.shape}")

        arrays = (self.current_frf, self.speed_frf, self.position_frf, self.context_vector)
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("FRF task contains NaN or infinite values")
        if self.total_points != 123:
            raise ValueError(f"unexpected task point count: {self.total_points}")

        for frequencies in (
            self.current_frf[:, 0],
            *self.speed_frf[:, :, 0],
            self.position_frf[:, 0],
        ):
            if not np.all(np.diff(frequencies) > 0):
                raise ValueError("log-frequency values must be strictly increasing")

        for features in (self.current_frf, self.speed_frf[:, :, :5], self.position_frf):
            if not np.allclose(features[..., 2] ** 2 + features[..., 3] ** 2, 1.0):
                raise ValueError("phase sine/cosine features are not on the unit circle")
            if not np.isin(features[..., 4], (0.0, 1.0)).all():
                raise ValueError("quality mask must be binary")

        expected_amplitude = self.speed_amplitudes_mA / self.speed_amplitudes_mA.max()
        if not np.allclose(self.speed_frf[:, :, 5], expected_amplitude[:, None]):
            raise ValueError("speed amplitude feature does not match speed_amplitudes_mA")

        rebuilt_context = np.concatenate(
            [self.current_frf.reshape(-1), self.speed_frf.reshape(-1), self.position_frf.reshape(-1)]
        )
        if not np.array_equal(self.context_vector, rebuilt_context):
            raise ValueError("context_vector is inconsistent with grouped FRF arrays")


def load_frf_task(project_root: Path, task_id: str = EXPECTED_TASK_ID) -> FRFTask:
    processed = project_root / "data" / "processed"
    task_path = processed / "frf_tasks.npz"
    manifest_path = processed / "frf_tasks_manifest.json"
    with np.load(task_path, allow_pickle=False) as data:
        stored_task_id = str(data["task_id"].item())
        if stored_task_id != task_id:
            raise KeyError(f"task {task_id!r} is not present in {task_path.name}")
        task = FRFTask(
            task_id=stored_task_id,
            current_frf=data["current_frf"].copy(),
            speed_frf=data["speed_frf"].copy(),
            position_frf=data["position_frf"].copy(),
            speed_amplitudes_mA=data["speed_amplitudes_mA"].copy(),
            speed_roles=data["speed_roles"].copy(),
            speed_suspicious_mask=data["speed_suspicious_mask"].copy(),
            base_feature_names=tuple(str(value) for value in data["base_feature_names"]),
            speed_feature_names=tuple(str(value) for value in data["speed_feature_names"]),
            context_vector=data["context_vector"].copy(),
            manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
        )
    task.validate()
    return task
