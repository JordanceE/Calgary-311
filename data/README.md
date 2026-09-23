# Податоци

`raw/` ги содржи оригиналните агрегирани одговори од Calgary Open Data и Open-Meteo. Тие се создаваат со:

```bash
python code/download_data.py
```

Очекувани raw датотеки:

- `snow_daily_by_service.csv`
- `closure_grouped_2021_2025.csv`
- `community_service_name_2021_2025.csv`
- `community_agency_responsible_2021_2025.csv`
- `weather_era5_daily_2018_2025.csv`
- `weather_era5_daily_2018_2025.json`
- `download_manifest.json`

`processed/` се создава со `python code/run_analysis.py` и содржи дневни аналитички табели, model coefficients/importances, closure rates и cluster assignments.

Податоците се агрегирани: проектот не презема адреси или точни координати на поединечни 311 барања. Пред final push треба да се провери дали секоја датотека е под GitHub ограничувањето од 100 MB. Ако не е, треба да се користи Git LFS или да се договори друг начин на предавање.
