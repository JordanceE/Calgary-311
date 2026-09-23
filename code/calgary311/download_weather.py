"""Download and flatten the Open-Meteo ERA5 daily weather data."""

import json
import urllib.parse

from .api_client import request_json, write_rows


def download_weather_data(data_dir):
    print("Downloading ERA5 daily weather...")
    params = {
        "latitude": "51.0447", "longitude": "-114.0719",
        "start_date": "2018-01-01", "end_date": "2025-12-31",
        "daily": ",".join([
            "weather_code", "temperature_2m_mean", "temperature_2m_max",
            "temperature_2m_min", "apparent_temperature_min", "precipitation_sum",
            "rain_sum", "snowfall_sum", "precipitation_hours",
            "wind_speed_10m_max", "wind_gusts_10m_max",
        ]),
        "timezone": "America/Edmonton", "models": "era5",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    weather = request_json(url)
    (data_dir / "weather_era5_daily_2018_2025.json").write_text(
        json.dumps(weather, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    daily = weather["daily"]
    rows = [dict(zip(daily, values)) for values in zip(*(daily[key] for key in daily))]
    write_rows(data_dir / "weather_era5_daily_2018_2025.csv", rows)
