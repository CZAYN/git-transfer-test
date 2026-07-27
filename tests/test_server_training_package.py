import importlib.util
import json
from pathlib import Path
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SCRIPT = PROJECT_ROOT / "scripts" / "package_server_training.py"


def _load_packager():
    specification = importlib.util.spec_from_file_location(
        "package_server_training",
        PACKAGE_SCRIPT,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_server_package_is_reproducible_and_excludes_sealed_artifacts(tmp_path):
    packager = _load_packager()
    first_path = tmp_path / "first.zip"
    second_path = tmp_path / "second.zip"
    first = packager.build_package(PROJECT_ROOT, first_path, overwrite=False)
    second = packager.build_package(PROJECT_ROOT, second_path, overwrite=False)
    assert first["archive_sha256"] == second["archive_sha256"]

    with zipfile.ZipFile(first_path) as archive:
        names = set(archive.namelist())
        embedded = json.loads(
            archive.read("SERVER_PACKAGE_MANIFEST.json").decode("utf-8")
        )
    assert "scripts/train_sac.py" in names
    assert "config/sac_training_v1.json" in names
    assert "data/processed/physics_motor_ensemble_v1.npz" in names
    assert embedded["entry_count"] == len(names) - 1
    assert not any("physics_motor_test" in name for name in names)
    assert not any("final_test" in name for name in names)
