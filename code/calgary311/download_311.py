"""Download the aggregated Calgary 311 datasets used by Q1–Q4."""

from .api_client import socrata_rows, write_rows

SNOW_SERVICES = [
    "Bylaw - Snow and Ice on Sidewalk",
    "Roads - Snow and Ice Control",
    "Parks - Snow and Ice Concerns - WAM",
    "Z - Roads - Snow & Ice Control",
    "CT - Snow and Ice Control",
    "Roads - Pathway Snow and Ice Concerns",
    "Parks - Snow and Ice Concerns - GIS",
]


def _download_if_missing(path, rows):
    """Keep completed downloads and replace only missing/empty files."""
    if path.exists() and path.stat().st_size > 0:
        print(f"Using existing {path}")
        return
    write_rows(path, rows)


def _community_rows(dimension):
    """Query one year at a time so Socrata does not time out on a large group-by."""
    for year in range(2021, 2026):
        print(f"  {dimension}: {year}")
        group = f"year, comm_code, comm_name, {dimension}"
        yield from socrata_rows({
            "$select": f"date_extract_y(requested_date) as year, comm_code, comm_name, {dimension}, count(*) as n",
            "$where": (
                f"requested_date between '{year}-01-01T00:00:00' and '{year}-12-31T23:59:59' "
                "and comm_code is not null and not status_description like 'Duplicate%' "
                "and status_description != 'TO BE DELETED'"
            ),
            "$group": group,
            "$order": group,
        }, page_size=50000)


def download_311_data(data_dir):
    quoted = ",".join("'" + item.replace("'", "''") + "'" for item in SNOW_SERVICES)
    print("Downloading daily snow/ice request counts...")
    _download_if_missing(
        data_dir / "snow_daily_by_service.csv",
        socrata_rows({
            "$select": "requested_date, service_name, count(*) as request_count",
            "$where": (
                "requested_date between '2018-01-01T00:00:00' and '2025-12-31T00:00:00' "
                f"and service_name in ({quoted})"
            ),
            "$group": "requested_date, service_name",
            "$order": "requested_date, service_name",
        }),
    )

    print("Downloading grouped closure outcomes...")
    closure_select = ", ".join([
        "date_extract_y(requested_date) as year",
        "date_extract_m(requested_date) as month",
        "date_extract_dow(requested_date) as day_of_week",
        "source", "service_name", "agency_responsible", "location_type",
        "count(*) as n",
        "sum(case when closed_date is not null and date_diff_d(requested_date, closed_date) <= 7 then 1 else 0 end) as quick",
    ])
    closure_group = "year, month, day_of_week, source, service_name, agency_responsible, location_type"
    _download_if_missing(
        data_dir / "closure_grouped_2021_2025.csv",
        socrata_rows({
            "$select": closure_select,
            "$where": (
                "requested_date between '2021-01-01T00:00:00' and '2025-12-31T00:00:00' "
                "and not status_description like 'Duplicate%' and status_description != 'TO BE DELETED'"
            ),
            "$group": closure_group,
            "$order": closure_group,
        }, page_size=25000),
    )

    for dimension in ("service_name", "agency_responsible"):
        print(f"Downloading community profiles by {dimension}...")
        _download_if_missing(
            data_dir / f"community_{dimension}_2021_2025.csv",
            _community_rows(dimension),
        )
