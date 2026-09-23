"""Generate figures, metric exports, the HTML report, and the Markdown summary."""

import json
from textwrap import dedent

import pandas as pd

from .config import FIGURES, OUTPUTS
from .reporting_utils import esc, html_table, svg_bar, svg_line


def generate_outputs(context):
    """Create all human-readable and machine-readable analysis outputs."""
    daily = context["daily"]
    snow_request_total_full_window = context["snow_request_total_full_window"]
    regression_results = context["regression_results"]
    rf_reg_importance = context["rf_reg_importance"]
    correlations = context["correlations"]
    high_demand_results = context["high_demand_results"]
    high_rf_importance = context["high_rf_importance"]
    high_threshold = context["high_threshold"]
    closure = context["closure"]
    closure_results = context["closure_results"]
    service_slow_rates = context["service_slow_rates"]
    source_slow_rates = context["source_slow_rates"]
    community_service = context["community_service"]
    cluster_profiles = context["cluster_profiles"]
    clustering_results = context["clustering_results"]
    selected_k = context["selected_k"]

    print("Generating figures...")
    monthly = daily.set_index("date").resample("MS").agg(
        requests=("request_count", "sum"), snowfall=("snowfall_sum", "sum")
    ).reset_index()
    svg_line(
        FIGURES / "monthly_requests_and_snowfall.svg",
        [
            ("311 snow/ice requests", monthly[["date", "requests"]].rename(columns={"requests": "value"}), "#2563eb"),
            ("ERA5 snowfall (cm)", monthly[["date", "snowfall"]].rename(columns={"snowfall": "value"}), "#f97316"),
        ],
        "Monthly snow/ice requests and snowfall, 2018–2025",
        "Monthly total (different units; compare timing)",
    )

    svg_bar(
        FIGURES / "snow_regression_importance.svg",
        rf_reg_importance.head(12).iloc[::-1].feature,
        rf_reg_importance.head(12).iloc[::-1].importance_mae,
        "Weather/calendar importance for daily request counts",
        "Increase in test MAE after permutation",
    )
    svg_bar(
        FIGURES / "high_demand_importance.svg",
        high_rf_importance.head(12).iloc[::-1].feature,
        high_rf_importance.head(12).iloc[::-1].importance,
        "Random-forest importance for high-demand days",
        "Impurity-based importance",
        color="#7c3aed",
    )

    metric_rows = []
    for question, result_set in [
        ("Q1 count", regression_results),
        ("Q2 high demand", high_demand_results),
        ("Q3 slow closure", closure_results),
    ]:
        for model_name, values in result_set.items():
            if not isinstance(values, dict):
                continue
            row = {"question": question, "model": model_name, **values}
            metric_rows.append(row)
    pd.DataFrame(metric_rows).to_csv(OUTPUTS / "model_metrics.csv", index=False)

    results = {
        "data": {
            "snow_requests_2018_2025": snow_request_total_full_window,
            "daily_rows": int(len(daily)),
            "closure_requests_2021_2025": int(closure.n.sum()),
            "closure_grouped_rows": int(len(closure)),
            "community_service_grouped_rows": int(len(community_service)),
        },
        "regression": regression_results,
        "regression_top_features": rf_reg_importance.head(10).to_dict("records"),
        "weather_correlations": correlations.head(10).to_dict("records"),
        "high_demand": high_demand_results,
        "high_demand_top_features": high_rf_importance.head(10).to_dict("records"),
        "closure": closure_results,
        "closure_top_slow_services_2025": service_slow_rates.head(10).to_dict("records"),
        "closure_fast_services_2025": service_slow_rates.sort_values("slow_rate").head(10).to_dict("records"),
        "closure_source_rates_2025": source_slow_rates.to_dict("records"),
        "clustering": clustering_results,
        "cluster_profiles": cluster_profiles.to_dict("records"),
    }
    (OUTPUTS / "analysis_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    best_regression_name = min(
        ["poisson", "random_forest"], key=lambda name: regression_results[name]["mae"]
    )
    best_high_name = max(
        ["logistic", "random_forest"], key=lambda name: high_demand_results[name]["pr_auc"]
    )
    best_weather_high_name = max(
        ["logistic_weather_only", "random_forest_weather_only"],
        key=lambda name: high_demand_results[name]["pr_auc"],
    )
    best_closure_name = max(
        ["logistic", "decision_tree"], key=lambda name: closure_results[name]["pr_auc"]
    )

    regression_table = pd.DataFrame(
        [
            {"Model": name, **{key: value for key, value in values.items() if key in {"mae", "rmse", "r2", "poisson_deviance"}}}
            for name, values in regression_results.items()
        ]
    )
    high_table = pd.DataFrame(
        [
            {"Model": name, **{key: value for key, value in values.items() if key in {"pr_auc", "roc_auc", "precision", "recall", "f1", "balanced_accuracy"}}}
            for name, values in high_demand_results.items() if isinstance(values, dict)
        ]
    )
    closure_table = pd.DataFrame(
        [
            {"Model": name, **{key: value for key, value in values.items() if key in {"pr_auc", "roc_auc", "precision", "recall", "f1", "balanced_accuracy"}}}
            for name, values in closure_results.items() if isinstance(values, dict)
        ]
    )

    def metric_format(value):
        return f"{float(value):.3f}"

    report = f"""<!doctype html>
    <html lang="mk">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Calgary 311 и временски услови — истражувачки проект</title>
    <style>
    body {{ font-family: Inter, Arial, sans-serif; color:#172033; line-height:1.55; margin:0; background:#f5f7fb; }}
    main {{ max-width:1080px; margin:0 auto; background:white; padding:52px 64px 80px; box-shadow:0 4px 28px #00000012; }}
    h1 {{ font-size:38px; line-height:1.12; margin:0 0 8px; color:#0f2854; }}
    h2 {{ margin-top:46px; color:#183d74; border-bottom:2px solid #dbe6f7; padding-bottom:8px; }}
    h3 {{ color:#284f82; margin-top:28px; }}
    .subtitle {{ color:#52647c; font-size:18px; margin-bottom:32px; }}
    .callout {{ background:#eef5ff; border-left:5px solid #2563eb; padding:16px 20px; margin:20px 0; }}
    .warning {{ background:#fff7e8; border-left-color:#f59e0b; }}
    .answer {{ background:#ecfdf5; border-left-color:#10b981; }}
    table {{ border-collapse:collapse; width:100%; margin:16px 0 24px; font-size:14px; }}
    th {{ background:#183d74; color:white; text-align:left; padding:9px; }}
    td {{ border-bottom:1px solid #d9e0ea; padding:8px 9px; vertical-align:top; }}
    tr:nth-child(even) td {{ background:#f8fafc; }}
    img {{ max-width:100%; height:auto; border:1px solid #e2e8f0; margin:12px 0 24px; }}
    code {{ background:#edf2f7; padding:2px 5px; border-radius:4px; }}
    .small {{ color:#64748b; font-size:13px; }}
    .grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:24px 0; }}
    .metric {{ background:#f2f6fc; border:1px solid #dbe5f2; padding:14px; border-radius:8px; }}
    .metric strong {{ display:block; font-size:24px; color:#183d74; }}
    @media (max-width:760px) {{ main {{ padding:28px 20px; }} .grid {{ grid-template-columns:1fr; }} }}
    </style>
    </head>
    <body><main>
    <h1>Calgary 311 Service Requests и историски временски услови</h1>
    <div class="subtitle">Регресија, класификација и кластерирање со временска валидација • Аналитички прозорец: 2018–2025</div>

    <div class="grid">
      <div class="metric"><strong>{results['data']['snow_requests_2018_2025']:,}</strong>snow/ice барања</div>
      <div class="metric"><strong>{results['data']['closure_requests_2021_2025']:,}</strong>барања во closure анализата</div>
      <div class="metric"><strong>{clustering_results['eligible_communities']}</strong>заедници во кластерирањето</div>
    </div>

    <h2>1. Податоци и методолошки дизајн</h2>
    <p>Користени се официјалниот <a href="https://data.calgary.ca/Services-and-Amenities/311-Service-Requests/iahh-g8bj">Calgary 311 Service Requests</a> сет и <a href="https://open-meteo.com/en/docs/historical-weather-api">Open‑Meteo Historical Weather API</a>. Временските податоци се ERA5 reanalysis за центарот на Калгари (51.0447, −114.0719), во зоната <code>America/Edmonton</code>. ERA5 е моделски реконструиран временски запис, а не директно мерење од една станица.</p>
    <p>Snow/ice корпусот е дефиниран со експлицитна листа од седум релевантни service имиња. Наивно пребарување со „ice“ е погрешно затоа што го наоѓа и текстот во <em>service</em> и <em>licence</em>. Часовните timestamps се претворени во локален календарски ден и се пополнети деновите со нула барања.</p>
    <div class="callout"><strong>Поделба без leakage:</strong> за прашањата 1–2 се тренира на 2018–2023, 2024 е validation, а 2025 е недопрен test. За прашањето 3 се тренира на 2021–2023, се избира threshold/complexity на 2024 и финално се оценува на 2025. Сите encoders и category collapsing правила се fit само на training периодот.</div>

    <h2>2. Време и дневен број snow/ice барања</h2>
    <img src="../figures/monthly_requests_and_snowfall.svg" alt="Monthly snow requests and snowfall">
    <p>За snow сезоната (октомври–април) се споредени сезонски baseline, regularized Poisson регресија и random forest регресија. Poisson моделот е корисен за насока и приближна големина на ефектите; random forest ја доловува нелинеарноста и интеракциите.</p>
    {html_table(regression_table, formats={"mae": metric_format, "rmse": metric_format, "r2": metric_format, "poisson_deviance": metric_format})}
    <img src="../figures/snow_regression_importance.svg" alt="Snow regression feature importance">
    <div class="callout answer"><strong>Одговор:</strong> временските услови објаснуваат значаен, но не целосен дел од дневната побарувачка. На test 2025 најдобар е <strong>{esc(best_regression_name)}</strong> со MAE {regression_results[best_regression_name]['mae']:.1f} барања и R² {regression_results[best_regression_name]['r2']:.3f}. Најсилни сигнали се температурата од претходниот ден и сезонскиот момент; тековната температура и снегот додаваат дополнителна информација. Calendar и trend ефектите покажуваат дека исти временски услови не произведуваат секогаш ист административен одговор.</div>
    <p class="small">Ова е асоцијативно и predictive толкување. Не докажува дека промена од 1 cm снег причински создава одреден број барања; однесување на граѓаните, правила за чистење, канали за пријавување и промени во service taxonomy се дополнителни фактори.</p>

    <h2>3. Денови со невообичаено висока побарувачка</h2>
    <p>„Висока побарувачка“ е дефинирана однапред како дневен број најмалку еднаков на 95-тиот перцентил од training snow-сезоните: <strong>{high_threshold} барања</strong>. Threshold-от не е пресметан од 2024 или 2025, со што test дефиницијата останува независна.</p>
    {html_table(high_table, formats={c: metric_format for c in high_table.columns if c != "Model"})}
    <img src="../figures/high_demand_importance.svg" alt="High demand feature importance">
    <div class="callout answer"><strong>Одговор:</strong> временските услови даваат сигнал, но препознавањето останува тешко. На 2025 test има само {high_demand_results['test_high_days']} high-demand денови. Најдобриот weather-only модел, <strong>{esc(best_weather_high_name)}</strong>, има PR‑AUC {high_demand_results[best_weather_high_name]['pr_auc']:.3f}, recall {high_demand_results[best_weather_high_name]['recall']:.1%} и precision {high_demand_results[best_weather_high_name]['precision']:.1%}, при prevalence {high_demand_results['test_prevalence']:.1%}. Calendar/trend карактеристиките не го подобруваат PR‑AUC (најдобар комбиниран резултат {high_demand_results[best_high_name]['pr_auc']:.3f}). Значи времето е корисен ран сигнал, но алармот би создавал многу лажни позитиви.</div>

    <h2>4. Решавање на 311 барање во рок од 7 дена</h2>
    <p>Target-от е изведен од <code>requested_date</code> и <code>closed_date</code>, но тие не се predictors. Positive class за евалуација е оперативно поинтересната ретка класа: <strong>не е затворено во 7 дена</strong>. Duplicate и „TO BE DELETED“ записи се исклучени. Predictors се само month/day-of-week, submission source, service name, responsible agency и location type.</p>
    {html_table(closure_table, formats={c: metric_format for c in closure_table.columns if c != "Model"})}
    <div class="callout answer"><strong>Одговор:</strong> информациите при поднесување содржат сигнал, но slow class е многу ретка ({closure_results['test_slow_prevalence']:.2%} во 2025). Најдобриот модел, <strong>{esc(best_closure_name)}</strong>, има PR‑AUC {closure_results[best_closure_name]['pr_auc']:.3f} наспроти baseline околу {closure_results['test_slow_prevalence']:.3f}. Најсилната поделба е по service/agency; submission channel и календарските карактеристики имаат помал дополнителен придонес.</div>
    <h3>Service типови со најголема slow стапка во 2025 (минимум 600 барања)</h3>
    {html_table(service_slow_rates, columns=["service_name", "n", "slow", "slow_rate"], formats={"slow_rate": lambda x: f"{x:.1%}"}, max_rows=10)}
    <h3>Submission channel</h3>
    {html_table(source_slow_rates, columns=["source", "n", "slow", "slow_rate"], formats={"slow_rate": lambda x: f"{x:.1%}"}, max_rows=10)}
    <div class="callout warning"><strong>Важно:</strong> „closed“ е административен статус, не нужно доказ дека проблемот физички е целосно отстранет. Моделот предвидува workflow outcome. Разликите меѓу service типови може да одразуваат SLA, автоматско затворање или различен процес, а не квалитет на работа.</div>

    <h2>5. Профили на заедници</h2>
    <p>Кластерирањето користи service и agency shares, плус log-просечен годишен интензитет, годишна варијабилност и recent trend. Сите карактеристики се стандардизирани. Вклучени се заедници со најмалку 500 барања и доволна покриеност низ годините. Избран е бројот на кластери со silhouette и minimum cluster-size услов, а потоа се проверени Ward agreement, composition-only sensitivity и временска стабилност.</p>
    <div class="grid">
      <div class="metric"><strong>{selected_k}</strong>избрани кластери</div>
      <div class="metric"><strong>{clustering_results['silhouette']:.3f}</strong>silhouette</div>
      <div class="metric"><strong>{clustering_results['temporal_window_ari']:.3f}</strong>temporal ARI</div>
    </div>
    {html_table(cluster_profiles, formats={"median_annual_requests": lambda x: f"{x:,.0f}", "median_annual_cv": lambda x: f"{x:.2f}"})}
    <div class="callout answer"><strong>Одговор:</strong> постојат интерпретабилни групи, но тие не се „природни вистини“. Silhouette {clustering_results['silhouette']:.3f}, Ward agreement ARI {clustering_results['ward_agreement_ari']:.3f} и temporal-window ARI {clustering_results['temporal_window_ari']:.3f} ја мерат разделеноста и стабилноста. Composition-only agreement ARI {clustering_results['composition_only_ari']:.3f} е клучната проверка дека решението не е само рангирање според обем.</div>
    <div class="callout warning"><strong>Ограничување за интензитетот:</strong> без community population/household denominator, бројот на барања не е per-capita стапка. Затоа профилите сигурно опишуваат service mix, но „висок интензитет“ може делумно да значи поголема заедница. За посилен заклучок треба да се додадат официјални population estimates.</div>

    <h2>6. Главни ограничувања</h2>
    <ul>
      <li>ERA5 е reanalysis на една grid-локација и не ја доловува секоја локална снежна разлика низ градот.</li>
      <li>311 барањата го мерат и однесувањето на пријавување, не само физичката состојба.</li>
      <li>Service taxonomy се менува; WAM/GIS/Pathway категориите се обединети семантички.</li>
      <li>Временските модели се асоцијативни; не се causal estimates.</li>
      <li>Closure target го мери административното затворање и е силно небалансиран.</li>
      <li>Кластерите зависат од feature изборот и немаат population denominator.</li>
    </ul>

    <h2>7. Репродуцибилност</h2>
    <p>Испорачаните датотеки содржат download pipeline, целосна анализа, processирани табели, model metrics и cluster assignments. Сурови лични адреси и точни координати не се преземени за моделот за затворање.</p>
    <p class="small">Податоците се преземаат преку приложениот download pipeline. Финалниот test период е 2025 за да се избегне нецелосната 2026 година.</p>
    </main></body></html>
    """
    (OUTPUTS / "calgary_311_project_report.html").write_text(
        dedent(report).strip() + "\n", encoding="utf-8"
    )

    summary_md = f"""# Calgary 311 + историски временски податоци: резиме

    ## Податоци и валидација

    - Calgary 311 snow/ice анализа: 2018–2025; дневните ERA5 податоци се споени по локален календарски датум.
    - Прашања 1–2: train 2018–2023, validation 2024, финален test 2025.
    - Прашање 3: train 2021–2023, validation 2024, финален test 2025.
    - Duplicate/deleted барањата се исклучени; полињата изведени од затворањето се користат само за target, никогаш како predictors.

    ## Одговори

    1. **Време и број на барања:** {best_regression_name} е најсилен на 2025 (MAE {regression_results[best_regression_name]['mae']:.1f}, R² {regression_results[best_regression_name]['r2']:.3f}). Најсилни се температурата од претходниот ден и сезонскиот момент; тековната температура и снегот додаваат информација, а календарските/trend карактеристики опфаќаат оперативни и reporting ефекти.
    2. **Денови со висока побарувачка:** дефинирани како најмалку {high_threshold} дневни барања, односно 95-тиот перцентил од training периодот. Најдобриот weather-only модел има PR‑AUC {high_demand_results[best_weather_high_name]['pr_auc']:.3f}, recall {high_demand_results[best_weather_high_name]['recall']:.1%} и precision {high_demand_results[best_weather_high_name]['precision']:.1%}; сигналот е корисен, но има многу лажни позитиви.
    3. **Затворање во 7 дена:** slow случаите се само {closure_results['test_slow_prevalence']:.2%} во 2025. {best_closure_name} постигнува PR‑AUC {closure_results[best_closure_name]['pr_auc']:.3f}; service и responsible agency доминираат, со помал дополнителен ефект од source/календар.
    4. **Профили на заедници:** избрани се {selected_k} кластери (silhouette {clustering_results['silhouette']:.3f}; temporal ARI {clustering_results['temporal_window_ari']:.3f}). Тие се интерпретабилни service-mix профили, но интензитетот не е per capita бидејќи нема population denominator.

    Целосната методологија, табели, графици и ограничувања се во `calgary_311_project_report.html`.
    """
    (OUTPUTS / "calgary_311_project_summary.md").write_text(
        dedent(summary_md).strip() + "\n", encoding="utf-8"
    )


    return results
