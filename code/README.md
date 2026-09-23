# Организација на кодот

Кодот е поделен според одговорност за полесно следење и одбрана на проектот.

| Датотека | Одговорност |
|---|---|
| `download_data.py` | Entry point за сите преземања |
| `run_analysis.py` | Entry point за Q1–Q4 и финалните outputs |
| `calgary311/api_client.py` | HTTP retry, Socrata pagination и CSV запишување |
| `calgary311/download_311.py` | Snow/ice, closure и community 311 агрегати |
| `calgary311/download_weather.py` | ERA5 дневни временски податоци |
| `calgary311/snow_weather.py` | Дневна подготовка, count регресија и high-demand класификација |
| `calgary311/closure_model.py` | Подготовка и класификација за затворање во 7 дена |
| `calgary311/community_clusters.py` | Community features, K-means и stability checks |
| `calgary311/evaluation.py` | Заеднички regression/classification метрики |
| `calgary311/reporting.py` | CSV/JSON, графици и извештаи |
| `calgary311/pipeline.py` | Го контролира редоследот без да содржи modeling логика |

Главниот редослед е:

```text
download_data.py → data/raw
run_analysis.py → preparation/modeling → data/processed + outputs
```

Старите скрипти во коренот и во `work/` се compatibility wrappers и ги повикуваат истите модули.
