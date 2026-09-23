"""Community feature construction, clustering, and stability checks (Q4)."""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def run_community_clustering(raw, processed, random_state):
    """Build normalized community profiles and evaluate stable cluster solutions."""
    RAW = raw
    PROCESSED = processed
    RANDOM_STATE = random_state
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


    return {
        "community_service": community_service,
        "community_agency": community_agency,
        "community_features": community_features,
        "cluster_profiles": cluster_profiles,
        "cluster_selection": cluster_selection,
        "clustering_results": clustering_results,
        "selected_k": selected_k,
    }
