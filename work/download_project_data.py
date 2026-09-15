from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


DATA_DIR = Path("work/data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)
CALGARY_RESOURCE = "https://data.calgary.ca/resource/iahh-g8bj.json"


def request_json(url: str, retries: int = 4):
    last_error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Calgary311Research/1.0"})
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.load(response)
        except Exception as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to retrieve {url}") from last_error


def socrata_rows(params: dict[str, str], page_size: int = 50000):
    offset = 0
    while True:
        page_params = dict(params)
        page_params["$limit"] = str(page_size)
        page_params["$offset"] = str(offset)
        url = CALGARY_RESOURCE + "?" + urllib.parse.urlencode(page_params)
        page = request_json(url)
        if not page:
            break
        yield from page
        print(f"  fetched {offset + len(page):,} grouped rows")
        if len(page) < page_size:
            break
        offset += page_size


def write_rows(path: Path, rows):
    iterator = iter(rows)
    first = next(iterator, None)
    if first is None:
        raise RuntimeError(f"No rows returned for {path.name}")
    fieldnames = list(first)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(first)
        count = 1
        for row in iterator:
            writer.writerow(row)
            count += 1
    print(f"Wrote {count:,} rows to {path}")


snow_services = [
    "Bylaw - Snow and Ice on Sidewalk",
    "Roads - Snow and Ice Control",
    "Parks - Snow and Ice Concerns - WAM",
    "Z - Roads - Snow & Ice Control",
    "CT - Snow and Ice Control",
    "Roads - Pathway Snow and Ice Concerns",
    "Parks - Snow and Ice Concerns - GIS",
]
quoted_services = ",".join("'" + item.replace("'", "''") + "'" for item in snow_services)

print("Downloading daily snow/ice request counts...")
write_rows(
    DATA_DIR / "snow_daily_by_service.csv",
    socrata_rows(
        {
            "$select": "requested_date, service_name, count(*) as request_count",
            "$where": (
                "requested_date between '2018-01-01T00:00:00' and '2025-12-31T00:00:00' "
                f"and service_name in ({quoted_services})"
            ),
            "$group": "requested_date, service_name",
            "$order": "requested_date, service_name",
        }
    ),
)

print("Downloading grouped closure outcomes...")
closure_select = ", ".join(
    [
        "date_extract_y(requested_date) as year",
        "date_extract_m(requested_date) as month",
        "date_extract_dow(requested_date) as day_of_week",
        "source",
        "service_name",
        "agency_responsible",
        "location_type",
        "count(*) as n",
        (
            "sum(case when closed_date is not null "
            "and date_diff_d(requested_date, closed_date) <= 7 "
            "then 1 else 0 end) as quick"
        ),
    ]
)
closure_group = "year, month, day_of_week, source, service_name, agency_responsible, location_type"
write_rows(
    DATA_DIR / "closure_grouped_2021_2025.csv",
    socrata_rows(
        {
            "$select": closure_select,
            "$where": (
                "requested_date between '2021-01-01T00:00:00' and '2025-12-31T00:00:00' "
                "and not status_description like 'Duplicate%' "
                "and status_description != 'TO BE DELETED'"
            ),
            "$group": closure_group,
            "$order": closure_group,
        },
        page_size=25000,
    ),
)

for dimension in ("service_name", "agency_responsible"):
    print(f"Downloading community profiles by {dimension}...")
    group = f"year, comm_code, comm_name, {dimension}"
    write_rows(
        DATA_DIR / f"community_{dimension}_2021_2025.csv",
        socrata_rows(
            {
                "$select": (
                    f"date_extract_y(requested_date) as year, comm_code, comm_name, {dimension}, count(*) as n"
                ),
                "$where": (
                    "requested_date between '2021-01-01T00:00:00' and '2025-12-31T00:00:00' "
                    "and comm_code is not null "
                    "and not status_description like 'Duplicate%' "
                    "and status_description != 'TO BE DELETED'"
                ),
                "$group": group,
                "$order": group,
            },
            page_size=50000,
        ),
    )

print("Downloading ERA5 daily weather...")
weather_params = {
    "latitude": "51.0447",
    "longitude": "-114.0719",
    "start_date": "2018-01-01",
    "end_date": "2025-12-31",
    "daily": ",".join(
        [
            "weather_code",
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "apparent_temperature_min",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
            "precipitation_hours",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
        ]
    ),
    "timezone": "America/Edmonton",
    "models": "era5",
}
weather_url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(weather_params)
weather = request_json(weather_url)
(DATA_DIR / "weather_era5_daily_2018_2025.json").write_text(
    json.dumps(weather, ensure_ascii=False, indent=2), encoding="utf-8"
)
daily = weather["daily"]
rows = [dict(zip(daily, values)) for values in zip(*(daily[key] for key in daily))]
write_rows(DATA_DIR / "weather_era5_daily_2018_2025.csv", rows)

manifest = {
    "calgary_dataset_id": "iahh-g8bj",
    "calgary_analysis_window": ["2018-01-01", "2025-12-31"],
    "closure_window": ["2021-01-01", "2025-12-31"],
    "snow_services": snow_services,
    "weather_model": "ERA5",
    "weather_coordinates": {"latitude": 51.0447, "longitude": -114.0719},
    "weather_timezone": "America/Edmonton",
}
(DATA_DIR / "download_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("Download complete.")
