from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elc_rl.data_processing import build_processed_data  # noqa: E402


if __name__ == "__main__":
    manifest = build_processed_data(PROJECT_ROOT)
    print(f"processed {manifest['counts']['total_points']} FRF points")
