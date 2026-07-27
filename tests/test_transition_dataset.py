from pathlib import Path

import numpy as np

from elc_rl.transition_dataset import (
    generate_transition_dataset,
    validate_transition_archive,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_small_transition_archive_is_aligned_and_reproducible(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    first_manifest = generate_transition_dataset(
        PROJECT_ROOT,
        first,
        transitions_per_stage=2,
        seed=731,
        stages=("current",),
    )
    generate_transition_dataset(
        PROJECT_ROOT,
        second,
        transitions_per_stage=2,
        seed=731,
        stages=("current",),
    )
    assert first_manifest["transition_count"] == 2
    assert first.read_bytes() == second.read_bytes()
    validation = validate_transition_archive(first)
    assert validation["stage_counts"]["current"] == 2
    with np.load(first, allow_pickle=False) as data:
        assert data["observation__time_metrics"].shape[0] == 2
        assert np.all(data["action"][:, 3:8] == 0.0)
