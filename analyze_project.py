from __future__ import annotations

import html
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "work" / "data" / "raw"
PROCESSED = ROOT / "work" / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
PROCESSED.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 20260831


def weighted_threshold(y, probability, weight, metric="f1"):
    candidates = np.unique(np.quantile(probability, np.linspace(0.02, 0.98, 160)))
    best = (0.5, -np.inf)
    for threshold in candidates:
        predicted = probability >= threshold
        if metric == "f1":
            score = f1_score(y, predicted, sample_weight=weight, zero_division=0)
        else:
            score = balanced_accuracy_score(y, predicted, sample_weight=weight)
        if score > best[1]:
            best = (float(threshold), float(score))
    return best


def classification_metrics(y, probability, threshold, weight=None):
    predicted = probability >= threshold
    return {
        "roc_auc": float(roc_auc_score(y, probability, sample_weight=weight)),
        "pr_auc": float(average_precision_score(y, probability, sample_weight=weight)),
        "brier": float(brier_score_loss(y, probability, sample_weight=weight)),
        "precision": float(precision_score(y, predicted, sample_weight=weight, zero_division=0)),
        "recall": float(recall_score(y, predicted, sample_weight=weight, zero_division=0)),
        "f1": float(f1_score(y, predicted, sample_weight=weight, zero_division=0)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, predicted, sample_weight=weight)
        ),
        "threshold": float(threshold),
    }


def regression_metrics(y, prediction):
    prediction = np.maximum(np.asarray(prediction), 1e-8)
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "r2": float(r2_score(y, prediction)),
        "poisson_deviance": float(mean_poisson_deviance(y, prediction)),
    }


def esc(value):
    return html.escape(str(value))


def svg_bar(path, labels, values, title, x_label="", color="#2563eb", width=980, height=520):
    labels = list(labels)
    values = np.asarray(values, dtype=float)
    margin = {"left": 310, "right": 45, "top": 72, "bottom": 65}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    row_h = plot_h / max(len(labels), 1)
    lo = min(0.0, float(values.min(initial=0)))
    hi = max(0.0, float(values.max(initial=1)))
    if math.isclose(lo, hi):
        hi = lo + 1

    def x_scale(value):
        return margin["left"] + (value - lo) / (hi - lo) * plot_w

    zero = x_scale(0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{esc(title)}</text>',
        f'<line x1="{zero:.1f}" y1="{margin["top"]}" x2="{zero:.1f}" y2="{height-margin["bottom"]}" stroke="#94a3b8"/>',
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        y = margin["top"] + index * row_h + row_h * 0.18
        bar_h = row_h * 0.64
        x_value = x_scale(value)
        x = min(zero, x_value)
        bar_w = max(abs(x_value - zero), 1)
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="3"/>'
        )
        parts.append(
            f'<text x="{margin["left"]-10}" y="{y+bar_h*0.72:.1f}" text-anchor="end" font-family="Arial" font-size="13">{esc(label)}</text>'
        )
        anchor = "start" if value >= 0 else "end"
        dx = 7 if value >= 0 else -7
        parts.append(
            f'<text x="{x_value+dx:.1f}" y="{y+bar_h*0.72:.1f}" text-anchor="{anchor}" font-family="Arial" font-size="12">{value:.3g}</text>'
        )
    parts.append(
        f'<text x="{margin["left"]+plot_w/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="14">{esc(x_label)}</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_line(path, series, title, y_label, width=1040, height=460):
    margin = {"left": 78, "right": 30, "top": 65, "bottom": 60}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    all_dates = pd.concat([frame[["date"]] for _, frame, _ in series])["date"]
    all_values = np.concatenate([frame["value"].to_numpy(float) for _, frame, _ in series])
    x_min, x_max = all_dates.min(), all_dates.max()
    y_min = min(0.0, float(np.nanmin(all_values)))
    y_max = float(np.nanmax(all_values)) * 1.05

    def xs(date):
        return margin["left"] + (date - x_min).days / max((x_max - x_min).days, 1) * plot_w

    def ys(value):
        return margin["top"] + (y_max - value) / max(y_max - y_min, 1e-9) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{esc(title)}</text>',
    ]
    for tick in np.linspace(y_min, y_max, 6):
        y = ys(tick)
        parts.append(f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{width-margin["right"]}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{margin["left"]-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{tick:.0f}</text>')
    for year in range(x_min.year, x_max.year + 1):
        date = pd.Timestamp(year=year, month=1, day=1)
        if x_min <= date <= x_max:
            x = xs(date)
            parts.append(f'<text x="{x:.1f}" y="{height-28}" text-anchor="middle" font-family="Arial" font-size="12">{year}</text>')
    for name, frame, color in series:
        points = " ".join(f"{xs(row.date):.1f},{ys(row.value):.1f}" for row in frame.itertuples())
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" opacity="0.9"/>')
    legend_x = margin["left"]
    for index, (name, _, color) in enumerate(series):
        x = legend_x + index * 190
        parts.append(f'<line x1="{x}" y1="50" x2="{x+25}" y2="50" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<text x="{x+32}" y="55" font-family="Arial" font-size="13">{esc(name)}</text>')
    parts.append(f'<text transform="translate(20 {margin["top"]+plot_h/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14">{esc(y_label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def html_table(frame, columns=None, formats=None, max_rows=20):
    data = frame.copy()
    if columns:
        data = data[columns]
    data = data.head(max_rows)
    formats = formats or {}
    rows = ["<table><thead><tr>" + "".join(f"<th>{esc(c)}</th>" for c in data.columns) + "</tr></thead><tbody>"]
    for _, row in data.iterrows():
        cells = []
        for column, value in row.items():
            if pd.isna(value):
                rendered = ""
            elif column in formats:
                rendered = formats[column](value)
            else:
                rendered = str(value)
            cells.append(f"<td>{esc(rendered)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    rows.append("</tbody></table>")
    return "\n".join(rows)


print("Preparing snow/weather daily table...")
snow_raw = pd.read_csv(RAW / "snow_daily_by_service.csv")
snow_raw["date"] = pd.to_datetime(snow_raw["requested_date"]).dt.floor("D")
snow_daily = snow_raw.groupby("date", as_index=False)["request_count"].sum()
weather = pd.read_csv(RAW / "weather_era5_daily_2018_2025.csv")
weather["date"] = pd.to_datetime(weather.pop("time"))
daily = weather.merge(snow_daily, how="left", on="date")
daily["request_count"] = daily["request_count"].fillna(0).astype(int)
snow_request_total_full_window = int(daily.request_count.sum())
daily["year"] = daily.date.dt.year
daily["month"] = daily.date.dt.month
daily["day_of_week"] = daily.date.dt.dayofweek
daily["day_of_year"] = daily.date.dt.dayofyear
daily["weekend"] = (daily.day_of_week >= 5).astype(int)
daily["snow_season"] = daily.month.isin([10, 11, 12, 1, 2, 3, 4])
daily["freeze_thaw"] = (
    (daily.temperature_2m_min < 0) & (daily.temperature_2m_max > 0)
).astype(int)
daily["month_sin"] = np.sin(2 * np.pi * daily.month / 12)
daily["month_cos"] = np.cos(2 * np.pi * daily.month / 12)
daily["doy_sin"] = np.sin(2 * np.pi * daily.day_of_year / 365.25)
daily["doy_cos"] = np.cos(2 * np.pi * daily.day_of_year / 365.25)
daily["trend"] = daily.year - 2018
for variable in ["snowfall_sum", "precipitation_sum", "temperature_2m_mean"]:
    for lag in (1, 2, 3):
        daily[f"{variable}_lag{lag}"] = daily[variable].shift(lag)
daily["snowfall_3day"] = daily["snowfall_sum"].rolling(3, min_periods=1).sum()
daily["precipitation_3day"] = daily["precipitation_sum"].rolling(3, min_periods=1).sum()
daily = daily.dropna().reset_index(drop=True)
daily.to_csv(PROCESSED / "daily_snow_weather.csv", index=False)

weather_features = [
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
    "freeze_thaw",
    "snowfall_sum_lag1",
    "snowfall_sum_lag2",
    "snowfall_sum_lag3",
    "precipitation_sum_lag1",
    "temperature_2m_mean_lag1",
    "snowfall_3day",
    "precipitation_3day",
]
calendar_features = ["month_sin", "month_cos", "doy_sin", "doy_cos", "weekend", "trend"]
model_features = weather_features + calendar_features
snow_model = daily[daily.snow_season].copy()
train = snow_model[snow_model.year <= 2023]
validation = snow_model[snow_model.year == 2024]
test = snow_model[snow_model.year == 2025]

print("Fitting count-regression models...")
poisson_candidates = []
for alpha in (0.001, 0.01, 0.1, 1.0):
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", PoissonRegressor(alpha=alpha, max_iter=1000)),
        ]
    )
    model.fit(train[model_features], train.request_count)
    prediction = model.predict(validation[model_features])
    poisson_candidates.append((mean_absolute_error(validation.request_count, prediction), alpha, model))
_, poisson_alpha, poisson_model = min(poisson_candidates, key=lambda item: item[0])

rf_reg_candidates = []
for max_depth in (5, 9, None):
    for min_leaf in (3, 10, 25):
        model = RandomForestRegressor(
            n_estimators=350,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(train[model_features], train.request_count)
        prediction = model.predict(validation[model_features])
        rf_reg_candidates.append(
            (mean_absolute_error(validation.request_count, prediction), max_depth, min_leaf, model)
        )
_, rf_depth, rf_leaf, rf_reg_model = min(rf_reg_candidates, key=lambda item: item[0])

seasonal_means = train.groupby("month").request_count.mean()
baseline_prediction = test.month.map(seasonal_means).fillna(train.request_count.mean()).to_numpy()
regression_results = {
    "seasonal_baseline": regression_metrics(test.request_count, baseline_prediction),
    "poisson": regression_metrics(test.request_count, poisson_model.predict(test[model_features])),
    "random_forest": regression_metrics(test.request_count, rf_reg_model.predict(test[model_features])),
}
regression_results["poisson"]["alpha"] = poisson_alpha
regression_results["random_forest"].update({"max_depth": rf_depth, "min_samples_leaf": rf_leaf})

permutation = permutation_importance(
    rf_reg_model,
    test[model_features],
    test.request_count,
    scoring="neg_mean_absolute_error",
    n_repeats=40,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf_reg_importance = pd.DataFrame(
    {"feature": model_features, "importance_mae": permutation.importances_mean}
).sort_values("importance_mae", ascending=False)
rf_reg_importance.to_csv(PROCESSED / "snow_regression_feature_importance.csv", index=False)

poisson_scale = poisson_model.named_steps["scale"]
poisson_coef = poisson_model.named_steps["model"].coef_
poisson_effects = pd.DataFrame(
    {
        "feature": model_features,
        "coefficient_per_sd": poisson_coef,
        "multiplicative_effect_per_sd": np.exp(poisson_coef),
    }
).sort_values("coefficient_per_sd", ascending=False)
poisson_effects.to_csv(PROCESSED / "poisson_standardized_effects.csv", index=False)

correlations = []
for feature in weather_features:
    feature_rank = snow_model[feature].rank(method="average")
    request_rank = snow_model.request_count.rank(method="average")
    rho = float(np.corrcoef(feature_rank, request_rank)[0, 1])
    correlations.append({"feature": feature, "spearman_rho": rho})
correlations = pd.DataFrame(correlations).sort_values("spearman_rho", key=np.abs, ascending=False)
correlations.to_csv(PROCESSED / "snow_weather_correlations.csv", index=False)

print("Fitting high-demand classifiers...")
high_threshold = int(math.ceil(train.request_count.quantile(0.95)))
for frame in (train, validation, test):
    frame["high_demand"] = (frame.request_count >= high_threshold).astype(int)

logit_candidates = []
for c_value in (0.05, 0.2, 1.0, 5.0):
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(train[model_features], train.high_demand)
    probability = model.predict_proba(validation[model_features])[:, 1]
    logit_candidates.append((average_precision_score(validation.high_demand, probability), c_value, model))
_, high_logit_c, high_logit = max(logit_candidates, key=lambda item: item[0])

rf_class_candidates = []
for max_depth in (4, 8, None):
    for min_leaf in (3, 10, 25):
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(train[model_features], train.high_demand)
        probability = model.predict_proba(validation[model_features])[:, 1]
        rf_class_candidates.append(
            (average_precision_score(validation.high_demand, probability), max_depth, min_leaf, model)
        )
_, high_rf_depth, high_rf_leaf, high_rf = max(rf_class_candidates, key=lambda item: item[0])

high_demand_results = {}
for name, model in [("logistic", high_logit), ("random_forest", high_rf)]:
    val_probability = model.predict_proba(validation[model_features])[:, 1]
    threshold, _ = weighted_threshold(validation.high_demand, val_probability, np.ones(len(validation)))
    test_probability = model.predict_proba(test[model_features])[:, 1]
    high_demand_results[name] = classification_metrics(
        test.high_demand, test_probability, threshold
    )
high_demand_results["logistic"]["C"] = high_logit_c
high_demand_results["random_forest"].update(
    {"max_depth": high_rf_depth, "min_samples_leaf": high_rf_leaf}
)

# Weather-only sensitivity check: same modeling families and a separate validation-based fit.
weather_logit_candidates = []
for c_value in (0.05, 0.2, 1.0, 5.0):
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(train[weather_features], train.high_demand)
    probability = model.predict_proba(validation[weather_features])[:, 1]
    weather_logit_candidates.append(
        (average_precision_score(validation.high_demand, probability), c_value, model)
    )
_, weather_logit_c, weather_logit = max(weather_logit_candidates, key=lambda item: item[0])

weather_rf_candidates = []
for max_depth in (4, 8, None):
    for min_leaf in (3, 10, 25):
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(train[weather_features], train.high_demand)
        probability = model.predict_proba(validation[weather_features])[:, 1]
        weather_rf_candidates.append(
            (average_precision_score(validation.high_demand, probability), max_depth, min_leaf, model)
        )
_, weather_rf_depth, weather_rf_leaf, weather_rf = max(
    weather_rf_candidates, key=lambda item: item[0]
)
for name, model in [("logistic_weather_only", weather_logit), ("random_forest_weather_only", weather_rf)]:
    val_probability = model.predict_proba(validation[weather_features])[:, 1]
    threshold, _ = weighted_threshold(validation.high_demand, val_probability, np.ones(len(validation)))
    test_probability = model.predict_proba(test[weather_features])[:, 1]
    high_demand_results[name] = classification_metrics(
        test.high_demand, test_probability, threshold
    )
high_demand_results["logistic_weather_only"]["C"] = weather_logit_c
high_demand_results["random_forest_weather_only"].update(
    {"max_depth": weather_rf_depth, "min_samples_leaf": weather_rf_leaf}
)
high_demand_results["test_prevalence"] = float(test.high_demand.mean())
high_demand_results["test_high_days"] = int(test.high_demand.sum())
high_demand_results["training_threshold_requests"] = high_threshold

high_rf_importance = pd.DataFrame(
    {"feature": model_features, "importance": high_rf.feature_importances_}
).sort_values("importance", ascending=False)
high_rf_importance.to_csv(PROCESSED / "high_demand_feature_importance.csv", index=False)

print("Preparing grouped closure model...")
closure = pd.read_csv(RAW / "closure_grouped_2021_2025.csv")
closure["slow"] = closure.n - closure.quick
closure["quick_rate"] = closure.quick / closure.n

category_columns = ["source", "service_name", "agency_responsible", "location_type"]
for column in category_columns:
    closure[column] = closure[column].fillna("MISSING").astype(str)

closure_train = closure[closure.year <= 2023].copy()
closure_validation = closure[closure.year == 2024].copy()
closure_test = closure[closure.year == 2025].copy()

def weighted_top(frame, column, count, top_n):
    return set(frame.groupby(column)[count].sum().nlargest(top_n).index)

top_service = weighted_top(closure_train, "service_name", "n", 180)
top_agency = weighted_top(closure_train, "agency_responsible", "n", 70)

def collapse_categories(frame):
    frame = frame.copy()
    frame["service_name"] = frame.service_name.where(frame.service_name.isin(top_service), "OTHER_SERVICE")
    frame["agency_responsible"] = frame.agency_responsible.where(
        frame.agency_responsible.isin(top_agency), "OTHER_AGENCY"
    )
    return frame

closure_train = collapse_categories(closure_train)
closure_validation = collapse_categories(closure_validation)
closure_test = collapse_categories(closure_test)

closure_features = category_columns + ["month", "day_of_week"]

def expand_grouped(frame):
    parts = []
    quick = frame[frame.quick > 0].copy()
    quick["target_slow"] = 0
    quick["sample_weight"] = quick.quick
    slow = frame[frame.slow > 0].copy()
    slow["target_slow"] = 1
    slow["sample_weight"] = slow.slow
    parts.extend([quick, slow])
    return pd.concat(parts, ignore_index=True)

train_expanded = expand_grouped(closure_train)
validation_expanded = expand_grouped(closure_validation)
test_expanded = expand_grouped(closure_test)

class_totals = train_expanded.groupby("target_slow").sample_weight.sum()
total_weight = class_totals.sum()
class_factor = {label: total_weight / (2 * value) for label, value in class_totals.items()}
for frame in (train_expanded, validation_expanded, test_expanded):
    frame["balanced_weight"] = frame.sample_weight * frame.target_slow.map(class_factor)

preprocess = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), category_columns),
        ("num", StandardScaler(), ["month", "day_of_week"]),
    ]
)

logistic_closure = Pipeline(
    [
        ("preprocess", preprocess),
        (
            "model",
            LogisticRegression(
                C=0.5,
                max_iter=1200,
                solver="liblinear",
                random_state=RANDOM_STATE,
                tol=1e-4,
            ),
        ),
    ]
)
logistic_closure.fit(
    train_expanded[closure_features],
    train_expanded.target_slow,
    model__sample_weight=train_expanded.balanced_weight,
)

tree_candidates = []
for depth in (6, 10, 14):
    tree = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=depth,
                    min_weight_fraction_leaf=0.0004,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    tree.fit(
        train_expanded[closure_features],
        train_expanded.target_slow,
        model__sample_weight=train_expanded.balanced_weight,
    )
    val_probability = tree.predict_proba(validation_expanded[closure_features])[:, 1]
    score = average_precision_score(
        validation_expanded.target_slow,
        val_probability,
        sample_weight=validation_expanded.sample_weight,
    )
    tree_candidates.append((score, depth, tree))
_, closure_tree_depth, closure_tree = max(tree_candidates, key=lambda item: item[0])

closure_results = {
    "train_requests": int(train_expanded.sample_weight.sum()),
    "validation_requests": int(validation_expanded.sample_weight.sum()),
    "test_requests": int(test_expanded.sample_weight.sum()),
    "test_slow_prevalence": float(
        np.average(test_expanded.target_slow, weights=test_expanded.sample_weight)
    ),
}
for name, model in [("logistic", logistic_closure), ("decision_tree", closure_tree)]:
    val_probability = model.predict_proba(validation_expanded[closure_features])[:, 1]
    threshold, _ = weighted_threshold(
        validation_expanded.target_slow,
        val_probability,
        validation_expanded.sample_weight,
    )
    test_probability = model.predict_proba(test_expanded[closure_features])[:, 1]
    closure_results[name] = classification_metrics(
        test_expanded.target_slow,
        test_probability,
        threshold,
        test_expanded.sample_weight,
    )
closure_results["decision_tree"]["max_depth"] = closure_tree_depth

feature_names = logistic_closure.named_steps["preprocess"].get_feature_names_out()
coefficients = logistic_closure.named_steps["model"].coef_[0]
closure_coefficients = pd.DataFrame(
    {"feature": feature_names, "coefficient_for_slow": coefficients, "odds_ratio": np.exp(coefficients)}
).sort_values("coefficient_for_slow", ascending=False)
closure_coefficients.to_csv(PROCESSED / "closure_logistic_coefficients.csv", index=False)

def actual_rate_table(frame, column, min_n=1000):
    table = frame.groupby(column, as_index=False)[["n", "slow"]].sum()
    table["slow_rate"] = table.slow / table.n
    return table[table.n >= min_n].sort_values("slow_rate", ascending=False)

service_slow_rates = actual_rate_table(closure_test, "service_name", min_n=600)
agency_slow_rates = actual_rate_table(closure_test, "agency_responsible", min_n=1500)
source_slow_rates = actual_rate_table(closure_test, "source", min_n=1)
service_slow_rates.to_csv(PROCESSED / "closure_2025_service_rates.csv", index=False)
agency_slow_rates.to_csv(PROCESSED / "closure_2025_agency_rates.csv", index=False)
source_slow_rates.to_csv(PROCESSED / "closure_2025_source_rates.csv", index=False)

print("Clustering community profiles...")
community_service = pd.read_csv(RAW / "community_service_name_2021_2025.csv")
community_agency = pd.read_csv(RAW / "community_agency_responsible_2021_2025.csv")
for frame in (community_service, community_agency):
    frame["comm_code"] = frame.comm_code.astype(str)
    frame["comm_name"] = frame.comm_name.fillna(frame.comm_code).astype(str)

top_services_cluster = list(
    community_service.groupby("service_name").n.sum().nlargest(15).index
)
top_agencies_cluster = list(
    community_agency.groupby("agency_responsible").n.sum().nlargest(8).index
)

def build_community_features(year_start, year_end):
    service = community_service[community_service.year.between(year_start, year_end)].copy()
    agency = community_agency[community_agency.year.between(year_start, year_end)].copy()
    totals_by_year = service.groupby(["comm_code", "comm_name", "year"], as_index=False).n.sum()
    eligibility = totals_by_year.groupby(["comm_code", "comm_name"]).agg(
        total_requests=("n", "sum"),
        active_years=("year", "nunique"),
        mean_annual_requests=("n", "mean"),
        sd_annual_requests=("n", "std"),
    ).reset_index()
    years_required = max(3, year_end - year_start)
    eligibility = eligibility[
        (eligibility.total_requests >= 500) & (eligibility.active_years >= years_required)
    ].copy()
    eligible = set(eligibility.comm_code)
    service = service[service.comm_code.isin(eligible)]
    agency = agency[agency.comm_code.isin(eligible)]

    service["profile"] = service.service_name.where(
        service.service_name.isin(top_services_cluster), "Other services"
    )
    service_profile = service.groupby(["comm_code", "profile"]).n.sum().unstack(fill_value=0)
    service_profile = service_profile.div(service_profile.sum(axis=1), axis=0)
    service_profile.columns = ["service_share__" + str(column) for column in service_profile.columns]

    agency["profile"] = agency.agency_responsible.where(
        agency.agency_responsible.isin(top_agencies_cluster), "Other agencies"
    )
    agency_profile = agency.groupby(["comm_code", "profile"]).n.sum().unstack(fill_value=0)
    agency_profile = agency_profile.div(agency_profile.sum(axis=1), axis=0)
    agency_profile.columns = ["agency_share__" + str(column) for column in agency_profile.columns]

    yearly = totals_by_year.pivot_table(index="comm_code", columns="year", values="n", fill_value=0)
    first_years = [year for year in yearly.columns if year <= year_start + 1]
    last_years = [year for year in yearly.columns if year >= year_end - 1]
    intensity = eligibility.set_index("comm_code")
    intensity["log_mean_annual_requests"] = np.log1p(intensity.mean_annual_requests)
    intensity["annual_cv"] = (
        intensity.sd_annual_requests.fillna(0) / intensity.mean_annual_requests.clip(lower=1)
    )
    intensity["recent_log_ratio"] = np.log1p(yearly[last_years].mean(axis=1)) - np.log1p(
        yearly[first_years].mean(axis=1)
    )
    intensity = intensity[
        ["comm_name", "total_requests", "mean_annual_requests", "log_mean_annual_requests", "annual_cv", "recent_log_ratio"]
    ]
    features = service_profile.join(agency_profile, how="inner").join(intensity, how="inner")
    return features


community_features = build_community_features(2021, 2025)
community_features_early = build_community_features(2021, 2024)
community_features_late = build_community_features(2022, 2025)
feature_columns = [
    column for column in community_features.columns if column.startswith("service_share__") or column.startswith("agency_share__")
] + ["log_mean_annual_requests", "annual_cv", "recent_log_ratio"]
composition_columns = [
    column for column in feature_columns if column.startswith("service_share__") or column.startswith("agency_share__")
]

scaler = StandardScaler()
community_scaled = scaler.fit_transform(community_features[feature_columns])
cluster_selection = []
cluster_models = {}
for k in range(2, 9):
    model = KMeans(n_clusters=k, n_init=80, random_state=RANDOM_STATE)
    labels = model.fit_predict(community_scaled)
    counts = pd.Series(labels).value_counts()
    score = silhouette_score(community_scaled, labels)
    cluster_selection.append(
        {"k": k, "silhouette": score, "smallest_cluster": int(counts.min())}
    )
    cluster_models[k] = (model, labels)
cluster_selection = pd.DataFrame(cluster_selection)
eligible_k = cluster_selection[cluster_selection.smallest_cluster >= 8]
selected_k = int(
    (eligible_k if not eligible_k.empty else cluster_selection).sort_values(
        ["silhouette", "smallest_cluster"], ascending=False
    ).iloc[0].k
)
kmeans_model, cluster_labels = cluster_models[selected_k]
community_features["cluster"] = cluster_labels + 1

ward_labels = AgglomerativeClustering(n_clusters=selected_k, linkage="ward").fit_predict(
    community_scaled
)
ward_ari = adjusted_rand_score(cluster_labels, ward_labels)

composition_scaled = StandardScaler().fit_transform(community_features[composition_columns])
composition_labels = KMeans(
    n_clusters=selected_k, n_init=80, random_state=RANDOM_STATE
).fit_predict(composition_scaled)
composition_ari = adjusted_rand_score(cluster_labels, composition_labels)

common_temporal = sorted(
    set(community_features_early.index)
    & set(community_features_late.index)
    & set(community_features.index)
)
common_features = [column for column in feature_columns if column in community_features_early.columns and column in community_features_late.columns]
early_scaled = StandardScaler().fit_transform(community_features_early.loc[common_temporal, common_features])
late_scaled = StandardScaler().fit_transform(community_features_late.loc[common_temporal, common_features])
early_labels = KMeans(n_clusters=selected_k, n_init=80, random_state=RANDOM_STATE).fit_predict(
    early_scaled
)
late_labels = KMeans(n_clusters=selected_k, n_init=80, random_state=RANDOM_STATE).fit_predict(
    late_scaled
)
temporal_ari = adjusted_rand_score(early_labels, late_labels)

cluster_profiles = []
overall = community_features[composition_columns].mean()
for cluster_id, group in community_features.groupby("cluster"):
    differences = group[composition_columns].mean() - overall
    top_dimensions = differences.nlargest(4)
    cluster_profiles.append(
        {
            "cluster": int(cluster_id),
            "communities": int(len(group)),
            "median_annual_requests": float(group.mean_annual_requests.median()),
            "median_annual_cv": float(group.annual_cv.median()),
            "profile": "; ".join(
                f"{name.split('__', 1)[-1]} (+{value:.1%})" for name, value in top_dimensions.items()
            ),
            "examples": ", ".join(group.sort_values("total_requests", ascending=False).comm_name.head(5)),
        }
    )
cluster_profiles = pd.DataFrame(cluster_profiles).sort_values("cluster")
community_features.reset_index().to_csv(PROCESSED / "community_cluster_assignments.csv", index=False)
cluster_profiles.to_csv(PROCESSED / "community_cluster_profiles.csv", index=False)
cluster_selection.to_csv(PROCESSED / "cluster_k_selection.csv", index=False)

clustering_results = {
    "eligible_communities": int(len(community_features)),
    "selected_k": selected_k,
    "silhouette": float(
        cluster_selection.loc[cluster_selection.k == selected_k, "silhouette"].iloc[0]
    ),
    "ward_agreement_ari": float(ward_ari),
    "composition_only_ari": float(composition_ari),
    "temporal_window_ari": float(temporal_ari),
}

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
    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
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
<img src="figures/monthly_requests_and_snowfall.svg" alt="Monthly snow requests and snowfall">
<p>За snow сезоната (октомври–април) се споредени сезонски baseline, regularized Poisson регресија и random forest регресија. Poisson моделот е корисен за насока и приближна големина на ефектите; random forest ја доловува нелинеарноста и интеракциите.</p>
{html_table(regression_table, formats={"mae": metric_format, "rmse": metric_format, "r2": metric_format, "poisson_deviance": metric_format})}
<img src="figures/snow_regression_importance.svg" alt="Snow regression feature importance">
<div class="callout answer"><strong>Одговор:</strong> временските услови објаснуваат значаен, но не целосен дел од дневната побарувачка. На test 2025 најдобар е <strong>{esc(best_regression_name)}</strong> со MAE {regression_results[best_regression_name]['mae']:.1f} барања и R² {regression_results[best_regression_name]['r2']:.3f}. Најсилни сигнали се температурата од претходниот ден и сезонскиот момент; тековната температура и снегот додаваат дополнителна информација. Calendar и trend ефектите покажуваат дека исти временски услови не произведуваат секогаш ист административен одговор.</div>
<p class="small">Ова е асоцијативно и predictive толкување. Не докажува дека промена од 1 cm снег причински создава одреден број барања; однесување на граѓаните, правила за чистење, канали за пријавување и промени во service taxonomy се дополнителни фактори.</p>

<h2>3. Денови со невообичаено висока побарувачка</h2>
<p>„Висока побарувачка“ е дефинирана однапред како дневен број најмалку еднаков на 95-тиот перцентил од training snow-сезоните: <strong>{high_threshold} барања</strong>. Threshold-от не е пресметан од 2024 или 2025, со што test дефиницијата останува независна.</p>
{html_table(high_table, formats={c: metric_format for c in high_table.columns if c != "Model"})}
<img src="figures/high_demand_importance.svg" alt="High demand feature importance">
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
<p class="small">Изворите се пристапени на 31 август 2026. Финалниот test период е 2025 за да се избегне нецелосната 2026 година.</p>
</main></body></html>
"""
(OUTPUTS / "calgary_311_project_report.html").write_text(report, encoding="utf-8")

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
(OUTPUTS / "calgary_311_project_summary.md").write_text(summary_md, encoding="utf-8")

shutil.copy2(ROOT / "work" / "download_project_data.py", OUTPUTS / "download_project_data.py")
shutil.copy2(Path(__file__), OUTPUTS / "analyze_project.py")
print("Analysis complete.")
