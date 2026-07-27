from pathlib import Path

import numpy as np

from elc_rl.data_processing import _canonical_arrays, parse_original_workbook
from elc_rl.task_dataset import load_frf_task


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = PROJECT_ROOT / "data" / "original" / "OSN600 9# CGS转台开环频响.xlsx"


def test_parse_has_expected_point_counts():
    frame = parse_original_workbook(WORKBOOK)
    assert len(frame) == 123
    assert (frame["loop"] == "current").sum() == 18
    assert (frame["loop"] == "speed").sum() == 90
    assert (frame["loop"] == "position").sum() == 15


def test_canonical_values_are_finite_and_ordered():
    frame = parse_original_workbook(WORKBOOK)
    arrays = _canonical_arrays(frame)
    assert np.isfinite(arrays["input_vector_v1"]).all()
    assert arrays["current_frequency_hz"].tolist() == sorted(arrays["current_frequency_hz"].tolist())
    assert np.all(np.diff(arrays["speed_frequency_hz"], axis=1) > 0)
    assert np.all(np.diff(arrays["position_frequency_hz"]) > 0)
    assert np.all(np.abs(arrays["input_vector_v1_scaled"][::2]) <= 1.0)


def test_quality_rules_keep_ood_and_suspicious_points():
    frame = parse_original_workbook(WORKBOOK)
    ten = frame[(frame["loop"] == "speed") & (frame["amplitude_mA"] == 10)]
    fifty = frame[(frame["loop"] == "speed") & (frame["amplitude_mA"] == 50)]
    assert all("ood_amplitude" in flags for flags in ten["quality_flags"])
    assert "suspicious_first_point" in fifty.iloc[0]["quality_flags"]
    assert len(frame[frame["quality_ok"]]) == 123


def test_processed_task_contains_all_three_measured_loops():
    task = load_frf_task(PROJECT_ROOT)
    assert task.current_frf.shape == (18, 5)
    assert task.speed_frf.shape == (6, 15, 6)
    assert task.position_frf.shape == (15, 5)
    assert np.isfinite(task.context_vector).all()
