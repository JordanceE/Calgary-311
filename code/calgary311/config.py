"""Project paths and shared constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
RANDOM_STATE = 20260831


def ensure_directories() -> None:
    """Create generated-data and output directories when they do not exist."""
    for directory in (RAW, PROCESSED, OUTPUTS, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
