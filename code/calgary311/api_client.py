"""HTTP, Socrata pagination, and CSV persistence helpers."""

import csv
import json
import time
import urllib.parse
import urllib.request

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


def write_rows(path, rows):
    temporary_path = path.with_suffix(path.suffix + ".part")
    iterator = iter(rows)
    first = next(iterator, None)
    if first is None:
        raise RuntimeError(f"No rows returned for {path.name}")
    fieldnames = list(first)
    count = 0
    with temporary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(first)
        count = 1
        for row in iterator:
            writer.writerow(row)
            count += 1
    temporary_path.replace(path)
    print(f"Wrote {count:,} rows to {path}")
