from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from calgary311.download_pipeline import run_downloads

if __name__ == "__main__":
    run_downloads()
