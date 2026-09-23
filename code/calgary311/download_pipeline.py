"""Orchestrate all official data downloads and write their manifest."""

import json

from .config import RAW, ensure_directories
from .download_311 import SNOW_SERVICES, download_311_data
from .download_weather import download_weather_data


def run_downloads():
    ensure_directories()
    download_311_data(RAW)
    download_weather_data(RAW)
    manifest = {
        "calgary_dataset_id": "iahh-g8bj",
        "calgary_analysis_window": ["2018-01-01", "2025-12-31"],
        "closure_window": ["2021-01-01", "2025-12-31"],
        "snow_services": SNOW_SERVICES,
        "weather_model": "ERA5",
        "weather_coordinates": {"latitude": 51.0447, "longitude": -114.0719},
        "weather_timezone": "America/Edmonton",
    }
    (RAW / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Download complete.")
