from pathlib import Path

import numpy as np

from elc_rl.task_dataset import load_frf_task


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_task_shapes_and_schema():
    task = load_frf_task(PROJECT_ROOT)
    assert task.task_id == "cgs_turntable_001"
    assert task.current_frf.shape == (18, 5)
    assert task.speed_frf.shape == (6, 15, 6)
    assert task.position_frf.shape == (15, 5)
    assert task.context_vector.shape == (705,)
    assert task.total_points == 123
    assert task.base_feature_names == (
        "log_frequency",
        "magnitude_db",
        "sin_phase",
        "cos_phase",
        "quality_mask",
    )


def test_task_frequency_phase_and_amplitude_features():
    task = load_frf_task(PROJECT_ROOT)
    assert np.all(np.diff(task.current_frf[:, 0]) > 0)
    assert np.all(np.diff(task.speed_frf[:, :, 0], axis=1) > 0)
    assert np.all(np.diff(task.position_frf[:, 0]) > 0)
    assert np.allclose(task.current_frf[:, 2] ** 2 + task.current_frf[:, 3] ** 2, 1.0)
    assert np.allclose(task.speed_frf[:, :, 2] ** 2 + task.speed_frf[:, :, 3] ** 2, 1.0)
    assert np.allclose(task.position_frf[:, 2] ** 2 + task.position_frf[:, 3] ** 2, 1.0)
    assert np.allclose(
        task.speed_frf[:, :, 5],
        (task.speed_amplitudes_mA / task.speed_amplitudes_mA.max())[:, None],
    )


def test_task_retains_and_masks_suspicious_speed_point():
    task = load_frf_task(PROJECT_ROOT)
    assert task.speed_suspicious_mask.sum() == 1
    assert task.speed_suspicious_mask[1, 0] == 1
    assert task.speed_frf[1, 0, 4] == 0.0
    quality_values = np.concatenate(
        [task.current_frf[:, 4], task.speed_frf[:, :, 4].reshape(-1), task.position_frf[:, 4]]
    )
    assert quality_values.sum() == 122
    assert len(quality_values) == 123


def test_task_manifest_has_no_rl_labels():
    task = load_frf_task(PROJECT_ROOT)
    text = str(task.manifest).lower()
    assert "static three-loop frf context" in task.manifest["description"].lower()
    assert task.manifest["context_vector"]["slices"]["position_frf"] == [630, 705]
    assert "reward" not in task.manifest["arrays"]
    assert "transition" not in task.manifest["arrays"]
    assert "controller" not in " ".join(task.manifest["arrays"]).lower()
    assert "not rl transitions" in text
