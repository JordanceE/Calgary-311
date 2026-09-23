"""Preparation and modeling for research questions 1 and 2."""

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import classification_metrics, regression_metrics, weighted_threshold


def run_snow_weather_analysis(raw, processed, random_state):
    """Prepare daily weather/request data and answer Q1–Q2."""
    RAW = raw
    PROCESSED = processed
    RANDOM_STATE = random_state
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


    return {
        "daily": daily,
        "snow_request_total_full_window": snow_request_total_full_window,
        "regression_results": regression_results,
        "rf_reg_importance": rf_reg_importance,
        "poisson_effects": poisson_effects,
        "correlations": correlations,
        "high_demand_results": high_demand_results,
        "high_rf_importance": high_rf_importance,
        "high_threshold": high_threshold,
    }
