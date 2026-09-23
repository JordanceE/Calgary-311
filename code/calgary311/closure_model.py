"""Grouped request preparation and seven-day closure classification (Q3)."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .evaluation import classification_metrics, weighted_threshold


def run_closure_analysis(raw, processed, random_state):
    """Model the risk that a request is not closed within seven days."""
    RAW = raw
    PROCESSED = processed
    RANDOM_STATE = random_state
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


    return {
        "closure": closure,
        "closure_results": closure_results,
        "closure_coefficients": closure_coefficients,
        "service_slow_rates": service_slow_rates,
        "agency_slow_rates": agency_slow_rates,
        "source_slow_rates": source_slow_rates,
    }
