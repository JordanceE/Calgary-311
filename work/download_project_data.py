"""Compatibility entry point; prefer `python code/download_data.py`."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent if HERE.name == "work" else HERE
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from calgary311.download_pipeline import run_downloads

if __name__ == "__main__":
    run_downloads()
