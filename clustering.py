
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN
)
from sklearn.metrics import silhouette_score


# --------------------------------------------------
# 1. FEATURE PREPARATION
# --------------------------------------------------

def prepare_features(df, selected_features, weights=None):

    weights = weights or {}

    feature_data = []
    feature_weights = []

    for col in selected_features:

        if col not in df.columns:
            raise ValueError(f"Feature not found: {col}")

        series = df[col]

        if pd.api.types.is_numeric_dtype(series):

            values = pd.to_numeric(series, errors="coerce")

            median = values.median()

            if pd.isna(median):
                median = 0

            values = values.fillna(median)

            part = pd.DataFrame({col: values})

        else:

            values = series.fillna("Unknown").astype(str)

            part = pd.get_dummies(
                values,
                prefix=col,
                dtype=float
            )

        weight = max(0, float(weights.get(col, 1)))

        if weight == 0:
            continue

        feature_data.append(part)

        feature_weights.extend(
            [np.sqrt(weight)] * part.shape[1]
        )

    if not feature_data:
        raise ValueError("No usable features selected.")

    X = pd.concat(feature_data, axis=1)

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    # Remove constant columns
    X = X.loc[:, X.nunique() > 1]

    if X.empty:
        raise ValueError("Selected features have no variation.")

    # Standardize features
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Apply weights after standardization
    original_columns = pd.concat(
        feature_data,
        axis=1
    ).columns

    valid_columns = X.columns

    weight_map = dict(
        zip(original_columns, feature_weights)
    )

    final_weights = np.array([
        weight_map[col] for col in valid_columns
    ])

    X_scaled = X_scaled * final_weights

    return X_scaled


# --------------------------------------------------
# 2. SILHOUETTE SCORE
# --------------------------------------------------

def calculate_silhouette(X, labels):

    labels = np.asarray(labels)

    # Exclude DBSCAN noise points
    valid_mask = labels != -1

    valid_labels = labels[valid_mask]
    valid_X = X[valid_mask]

    unique_labels = np.unique(valid_labels)

    if (
        len(unique_labels) < 2
        or len(valid_labels) <= len(unique_labels)
    ):
        return None

    return round(
        float(
            silhouette_score(
                valid_X,
                valid_labels
            )
        ),
        4
    )


# --------------------------------------------------
# 3. CLUSTER PROFILE
# --------------------------------------------------

def generate_cluster_profiles(df, labels, features):

    working_df = df.copy()

    working_df["_cluster"] = [
        "Noise" if label == -1
        else f"Cluster {int(label) + 1}"
        for label in labels
    ]

    profiles = {}

    for cluster_name, group in working_df.groupby(
        "_cluster",
        sort=False
    ):

        profile = {}

        for feature in features:

            if pd.api.types.is_numeric_dtype(
                df[feature]
            ):

                values = pd.to_numeric(
                    group[feature],
                    errors="coerce"
                )

                profile[feature] = {
                    "mean": round(float(values.mean()), 2)
                    if not pd.isna(values.mean()) else None,

                    "median": round(float(values.median()), 2)
                    if not pd.isna(values.median()) else None,

                    "std": round(float(values.std()), 2)
                    if not pd.isna(values.std()) else None
                }

            else:

                profile[feature] = {
                    "distribution": {
                        str(k): int(v)
                        for k, v in group[feature]
                        .fillna("Unknown")
                        .value_counts()
                        .items()
                    }
                }

        profiles[cluster_name] = {
            "size": int(len(group)),
            "profile": profile
        }

    return profiles


# --------------------------------------------------
# 4. HCP ASSIGNMENTS
# --------------------------------------------------

def generate_hcp_assignments(df, labels):

    assignments = []

    for position, label in enumerate(labels):

        record = {
            "row_index": int(position),
            "cluster": (
                "Noise" if label == -1
                else f"Cluster {int(label) + 1}"
            )
        }

        # Include actual HCP identifier when available
        if "HCP_ID" in df.columns:
            record["HCP_ID"] = str(
                df.iloc[position]["HCP_ID"]
            )

        if "HCP_Name" in df.columns:
            record["HCP_Name"] = str(
                df.iloc[position]["HCP_Name"]
            )

        assignments.append(record)

    return assignments


# --------------------------------------------------
# 5. ELBOW METHOD AND K COMPARISON
# --------------------------------------------------

def evaluate_k_values(X, max_k=6):

    n_samples = len(X)

    upper_k = min(max_k, n_samples - 1)

    elbow_results = []

    for k in range(2, upper_k + 1):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X)

        inertia = float(model.inertia_)

        score = calculate_silhouette(X, labels)

        elbow_results.append({
            "k": k,
            "inertia": round(inertia, 4),
            "silhouette": score
        })

    # Find the elbow using maximum perpendicular
    # distance from the first-last inertia line.
    suggested_k = None

    if len(elbow_results) >= 3:

        x = np.array([
            item["k"] for item in elbow_results
        ], dtype=float)

        y = np.array([
            item["inertia"] for item in elbow_results
        ], dtype=float)

        x_normalized = (
            (x - x.min()) / (x.max() - x.min())
        )

        y_range = y.max() - y.min()

        if y_range > 0:

            y_normalized = (
                (y - y.min()) / y_range
            )

            distances = np.abs(
                (y_normalized[-1] - y_normalized[0])
                * x_normalized
                - (x_normalized[-1] - x_normalized[0])
                * y_normalized
                + x_normalized[-1] * y_normalized[0]
                - y_normalized[-1] * x_normalized[0]
            ) / np.sqrt(
                (y_normalized[-1] - y_normalized[0]) ** 2
                + (x_normalized[-1] - x_normalized[0]) ** 2
            )

            suggested_k = int(
                x[np.argmax(distances)]
            )

    best_silhouette_k = None

    valid_scores = [
        item for item in elbow_results
        if item["silhouette"] is not None
    ]

    if valid_scores:

        best_silhouette_k = max(
            valid_scores,
            key=lambda item: item["silhouette"]
        )["k"]

    return {
        "tested_values": elbow_results,
        "suggested_k_by_elbow": suggested_k,
        "best_k_by_silhouette": best_silhouette_k
    }


# --------------------------------------------------
# 6. MAIN CLUSTERING FUNCTION
# --------------------------------------------------

def run_clustering(
    df,
    selected_features,
    weights=None,
    n_clusters=3
):

    if len(df) < 4:
        raise ValueError(
            "At least 4 records are required."
        )

    # Prepare numerical feature matrix
    X = prepare_features(
        df,
        selected_features,
        weights
    )

    k = min(
        max(2, int(n_clusters)),
        len(X) - 1
    )

    # Elbow analysis
    elbow_analysis = evaluate_k_values(
        X,
        max_k=6
    )

    models = [

        (
            "K-Means",
            KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )
        ),

        (
            "Hierarchical",
            AgglomerativeClustering(
                n_clusters=k
            )
        ),

        (
            "DBSCAN",
            DBSCAN(
                eps=0.5,
                min_samples=5
            )
        )

    ]

    results = []

    # Store assignments for cross-model comparison
    all_assignments = {}

    # --------------------------------------------------
    # RUN ALL THREE ALGORITHMS
    # --------------------------------------------------

    for name, model in models:

        try:

            labels = model.fit_predict(X)

            unique_labels = np.unique(labels)

            noise_count = (
                int(np.sum(labels == -1))
                if name == "DBSCAN"
                else 0
            )

            score = calculate_silhouette(
                X,
                labels
            )

            counts = pd.Series(
                labels
            ).value_counts().sort_index()

            cluster_sizes = {}

            for cluster_id, count in counts.items():

                cluster_name = (
                    "Noise"
                    if cluster_id == -1
                    else f"Cluster {int(cluster_id) + 1}"
                )

                cluster_sizes[cluster_name] = int(count)

            assignments = [
                "Noise" if label == -1
                else f"Cluster {int(label) + 1}"
                for label in labels
            ]

            # Cluster profiles using original data
            profiles = generate_cluster_profiles(
                df,
                labels,
                selected_features
            )

            # HCP-level assignment details
            hcp_assignments = generate_hcp_assignments(
                df,
                labels
            )

            all_assignments[name] = assignments

            results.append({

                "method": name,

                "silhouette": score,

                "clusters": len(
                    set(labels) - {-1}
                ),

                "noise_points": noise_count,

                "cluster_sizes": cluster_sizes,

                "cluster_profiles": profiles,

                "assignments": assignments,

                "hcp_assignments": hcp_assignments

            })

        except Exception as e:

            results.append({

                "method": name,
                "silhouette": None,
                "clusters": 0,
                "noise_points": 0,
                "cluster_sizes": {},
                "cluster_profiles": {},
                "assignments": [],
                "hcp_assignments": [],
                "error": str(e)

            })

    # --------------------------------------------------
    # BEST MODEL BY SILHOUETTE
    # --------------------------------------------------

    valid_results = [
        result for result in results
        if result["silhouette"] is not None
    ]

    best_method = (
        max(
            valid_results,
            key=lambda item: item["silhouette"]
        )["method"]
        if valid_results else None
    )

    # --------------------------------------------------
    # CROSS-MODEL ASSIGNMENT COMPARISON
    # --------------------------------------------------

    assignment_comparison = {}

    if (
        "K-Means" in all_assignments
        and "Hierarchical" in all_assignments
    ):

        km = all_assignments["K-Means"]
        hc = all_assignments["Hierarchical"]

        # Cluster numbers may be different, so compare
        # co-membership rather than raw cluster labels.
        same_partition = all(
            (km[i] == km[j]) == (hc[i] == hc[j])
            for i in range(len(km))
            for j in range(i + 1, len(km))
        )

        assignment_comparison[
            "kmeans_hierarchical_same_partition"
        ] = same_partition

    # --------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------

    return {

        "methods": results,

        "best_method": best_method,

        "selected_features": selected_features,

        "elbow_analysis": elbow_analysis,

        "assignment_comparison": assignment_comparison

    }