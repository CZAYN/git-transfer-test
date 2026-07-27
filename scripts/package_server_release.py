from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import ModuleType
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load packager: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def build_release(project_root: Path, output: Path, overwrite: bool) -> dict[str, object]:
    root = project_root.resolve()
    destination = output.resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"server release already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    training_packager = _load_module(
        "package_server_training_for_release",
        root / "scripts" / "package_server_training.py",
    )
    final_test_packager = _load_module(
        "package_server_final_test_for_release",
        root / "scripts" / "package_server_final_test.py",
    )
    with tempfile.TemporaryDirectory() as temporary:
        temporary_root = Path(temporary)
        training_zip = temporary_root / "elc_rl_server_training_v1.zip"
        final_test_zip = temporary_root / "elc_rl_server_final_test_v1.zip"
        training_packager.build_package(root, training_zip, overwrite=False)
        final_test_packager.build_package(root, final_test_zip, overwrite=False)

        release_files = (
            (
                "training/elc_rl_server_training_v1.zip",
                training_zip.read_bytes(),
            ),
            (
                "final_test/elc_rl_server_final_test_v1.zip",
                final_test_zip.read_bytes(),
            ),
            ("SERVER_RELEASE.md", (root / "SERVER_RELEASE.md").read_bytes()),
        )
        entries = [
            {
                "path": name,
                "size_bytes": len(content),
                "sha256": _sha256_bytes(content),
            }
            for name, content in release_files
        ]
        internal_manifest = {
            "schema_version": 1,
            "package_kind": "physics_v1_single_upload_release",
            "workflow_policy": (
                "extract the training package immediately; keep the final-test "
                "package unopened until the unique candidate is selected and frozen"
            ),
            "entry_count": len(entries),
            "entries": entries,
        }
        temporary_zip = destination.with_name(f".{destination.name}.tmp")
        with zipfile.ZipFile(
            temporary_zip,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name, content in release_files:
                archive.writestr(_zip_info(name), content)
            archive.writestr(
                _zip_info("SERVER_RELEASE_MANIFEST.json"),
                json.dumps(
                    internal_manifest,
                    ensure_ascii=False,
                    indent=2,
                ).encode("utf-8")
                + b"\n",
            )
        temporary_zip.replace(destination)

    external_manifest = {
        **internal_manifest,
        "archive": destination.name,
        "archive_size_bytes": destination.stat().st_size,
        "archive_sha256": _sha256_file(destination),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = destination.with_name(f"{destination.stem}_manifest.json")
    manifest_path.write_text(
        json.dumps(external_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return external_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the single-upload server release bundle."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    output = (
        root / "dist" / "elc_rl_server_release_v1.zip"
        if arguments.output is None
        else arguments.output.resolve()
    )
    result = build_release(root, output, arguments.overwrite)
    print(
        json.dumps(
            {
                "archive": str(output),
                "archive_size_bytes": result["archive_size_bytes"],
                "archive_sha256": result["archive_sha256"],
                "entry_count": result["entry_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
