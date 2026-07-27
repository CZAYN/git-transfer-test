import json
from pathlib import Path

import numpy as np
import pytest

from elc_rl.sac_training import (
    CandidatePool,
    CandidateRecord,
    _save_resume_checkpoint,
    build_training_input_manifest,
    load_formal_training_config,
    select_multi_seed_candidate,
)
from elc_rl.controller_parameters import load_physics_controller_parameter_space
from elc_rl.tuning_env import STAGE_ORDER


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_formal_training_configuration_is_complete_and_multi_seed():
    config = load_formal_training_config(PROJECT_ROOT)
    assert tuple(stage.name for stage in config.stages) == STAGE_ORDER
    assert all(stage.total_timesteps > 0 for stage in config.stages)
    assert len(config.seeds) >= 3
    assert len(set(config.seeds)) == len(config.seeds)
    assert config.payload["default_device"] == "cuda"
    assert config.payload["checkpoint"]["save_replay_buffer"]
    assert config.payload["tensorboard"]["enabled"]


def test_training_input_manifest_is_deterministic_and_training_only():
    config = load_formal_training_config(PROJECT_ROOT)
    first = build_training_input_manifest(PROJECT_ROOT, config)
    second = build_training_input_manifest(PROJECT_ROOT, config)
    assert first == second
    assert len(first["fingerprint"]) == 64
    names = {item["path"] for item in first["files"]}
    assert "data/processed/physics_motor_ensemble_v1.npz" in names
    assert "scripts/train_sac.py" in names
    assert not any("physics_motor_test" in name for name in names)
    assert not any("final_test" in name for name in names)


def test_candidate_pool_deduplicates_and_keeps_the_lowest_costs():
    pool = CandidatePool("joint", maximum_size=2)
    first = np.arange(11, dtype=np.float64)
    second = first + 1.0
    third = first + 2.0
    pool.add(CandidateRecord("joint", 3.0, first, 1))
    pool.add(CandidateRecord("joint", 2.0, second, 2))
    pool.add(CandidateRecord("joint", 1.0, third, 3))
    assert [record.fast_cost for record in pool.records] == [1.0, 2.0]

    pool.add(CandidateRecord("joint", 0.5, second.copy(), 4))
    assert len(pool.records) == 2
    assert [record.fast_cost for record in pool.records] == [0.5, 1.0]
    assert pool.records[0].global_timestep == 4


class _FakeSAC:
    def __init__(self, timesteps: int) -> None:
        self.num_timesteps = timesteps

    def save(self, path: Path) -> None:
        Path(path).write_bytes(b"model")

    def save_replay_buffer(self, path: Path) -> None:
        Path(path).write_bytes(b"replay")


def test_resume_checkpoint_is_complete_and_rotated(tmp_path):
    config = load_formal_training_config(PROJECT_ROOT)
    state = {
        "stage_index": 0,
        "stage": "current",
        "stage_timesteps_completed": 0,
        "global_timesteps_completed": 0,
        "config_sha256": config.sha256,
        "input_fingerprint": "a" * 64,
        "latest_checkpoint": None,
    }
    for steps in (10, 20, 30):
        state["stage_timesteps_completed"] = steps
        state["global_timesteps_completed"] = steps
        checkpoint = _save_resume_checkpoint(
            _FakeSAC(steps),
            tmp_path,
            state,
            config,
        )
        assert (checkpoint / "COMPLETE").is_file()
        assert (checkpoint / "model.zip").read_bytes() == b"model"
        assert (checkpoint / "replay_buffer.pkl").read_bytes() == b"replay"
        assert (checkpoint / "rng_state.pt").is_file()

    completed = [
        path
        for path in (tmp_path / "checkpoints").iterdir()
        if path.is_dir() and (path / "COMPLETE").is_file()
    ]
    assert len(completed) == config.payload["checkpoint"]["keep_last"]
    persisted = json.loads(
        (tmp_path / "trainer_state.json").read_text(encoding="utf-8")
    )
    assert persisted["global_timesteps_completed"] == 30
    assert (tmp_path / persisted["latest_checkpoint"]).is_dir()


def test_engineering_candidate_is_rejected_by_formal_selection(tmp_path):
    space = load_physics_controller_parameter_space(PROJECT_ROOT)
    candidate = tmp_path / "engineering_candidate.npz"
    np.savez_compressed(
        candidate,
        parameter_names=np.asarray(space.names),
        parameters=space.initial,
        seed=np.asarray(1, dtype=np.int64),
        training_complete=np.asarray(True),
        eligible_for_selection=np.asarray(False),
        input_fingerprint=np.asarray("a" * 64),
    )
    with pytest.raises(ValueError, match="not eligible"):
        select_multi_seed_candidate(
            PROJECT_ROOT,
            [candidate],
            tmp_path / "selection",
        )
