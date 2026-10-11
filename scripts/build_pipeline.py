"""End-to-end pipeline: raw CSVs -> processed parquet/JSON artifacts consumed by the Streamlit app.

Run with: python scripts/build_pipeline.py
This script trains nothing at app runtime -- it is the offline half of the
train/serve split described in the README.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import clustering, config, data_loader, features, segmentation


def log(msg: str) -> None:
    print(f"[build_pipeline] {msg}")


def main() -> None:
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    log("Loading raw tables and validating referential integrity...")
    tables, quality_report = data_loader.load_and_validate()
    log(f"data quality report: {quality_report}")

    log("Building seller-order pairs and seller-level features...")
    pairs, seller_features, reference_date = features.build_feature_table(tables)
    pairs.to_parquet(config.PROCESSED_FILES["seller_order_pairs"], index=False)
    log(f"seller_order_pairs: {len(pairs):,} rows")
    log(f"reference date (max order_purchase_timestamp): {reference_date}")

    multi_seller_stats = features.multi_seller_order_stats(tables["order_items"])
    log(f"multi-seller order stats: {multi_seller_stats}")

    sensitivity = features.threshold_sensitivity(seller_features)
    log(f"threshold sensitivity:\n{sensitivity}")

    eligible, excluded = features.apply_min_order_threshold(seller_features)
    log(f"eligible for clustering: {len(eligible):,} / excluded (low data): {len(excluded):,}")

    # ---- correlation diagnostic (drives the feature-set decision below) ----
    corr_matrix = eligible[config.CANDIDATE_FEATURES].corr(method="pearson").round(3)
    log(f"feature correlation matrix, candidate set (eligible population):\n{corr_matrix}")

    # ---- k-selection diagnostics ----
    X_eligible = eligible[config.CLUSTER_FEATURES].to_numpy()
    log("Running elbow/silhouette across k=2..10...")
    elbow_df = clustering.elbow_silhouette(X_eligible)
    log(f"elbow/silhouette:\n{elbow_df}")

    log(f"Running stability check (ARI across seeds) at k={config.N_CLUSTERS}...")
    stability_df = clustering.stability_ari(X_eligible, k=config.N_CLUSTERS)
    log(f"stability ARI:\n{stability_df}")

    # ---- DBSCAN critical comparison ----
    log("Computing k-distance plot for DBSCAN eps estimation...")
    kth_distances = clustering.k_distance_data(X_eligible)
    eps_estimate = clustering.estimate_eps_from_knee(kth_distances)
    log(f"estimated eps from knee: {eps_estimate:.3f}")
    dbscan_result = clustering.run_dbscan(X_eligible, eps=eps_estimate)
    dbscan_labels = dbscan_result.pop("labels")
    log(f"DBSCAN result (knee eps): {dbscan_result}")

    log("Sweeping eps around the knee to show the merge/fragment transition...")
    dbscan_sweep = []
    for mult in config.DBSCAN_EPS_MULTIPLIERS:
        r = clustering.run_dbscan(X_eligible, eps=eps_estimate * mult)
        r.pop("labels")
        r["eps_multiplier"] = mult
        dbscan_sweep.append(r)
    log(f"DBSCAN eps sweep:\n{pd.DataFrame(dbscan_sweep)}")

    # ---- PCA (visualization only) ----
    log("Computing PCA(2) projection for visualization...")
    pca_coords, explained_var = clustering.pca_projection(X_eligible)
    log(f"PCA explained variance ratio: {explained_var}, sum={explained_var.sum():.3f}")

    # ---- final model, health score, segment naming ----
    log(f"Fitting final K-Means (k={config.N_CLUSTERS}) and assigning segments...")
    result = segmentation.segment_sellers(seller_features)
    log(f"cluster profile table:\n{result.profile_table}")
    log(f"cluster -> segment key mapping: {result.cluster_to_key}")

    # PCA coordinates and DBSCAN labels exist only for the clustered
    # population; joined by seller_id so row order can never misalign them.
    diagnostics_cols = pd.DataFrame(
        {
            "seller_id": eligible["seller_id"].to_numpy(),
            "pc1": pca_coords[:, 0],
            "pc2": pca_coords[:, 1],
            "dbscan_label": dbscan_labels,
            "dbscan_is_noise": dbscan_labels == -1,
        }
    )
    all_sellers = result.sellers.merge(diagnostics_cols, on="seller_id", how="left")
    all_sellers["dbscan_is_noise"] = all_sellers["dbscan_is_noise"].eq(True)

    all_sellers.to_parquet(config.PROCESSED_FILES["sellers_segmented"], index=False)
    seller_features.to_parquet(config.PROCESSED_FILES["seller_features"], index=False)
    log(f"sellers_segmented.parquet written: {len(all_sellers):,} rows")

    with open(config.PROCESSED_FILES["cluster_profiles"], "w") as f:
        json.dump(segmentation.build_profile_records(result), f, indent=2)
    log("cluster_profiles.json written")

    # ---- model_diagnostics.json (Segmentation / Methodology tabs) ----
    diagnostics = {
        "reference_date": str(reference_date.date()),
        "min_orders_threshold": config.MIN_ORDERS_THRESHOLD,
        "n_clusters": config.N_CLUSTERS,
        "seed": config.SEED,
        "cluster_features": config.CLUSTER_FEATURES,
        "candidate_features": config.CANDIDATE_FEATURES,
        "n_sellers_total": int(len(all_sellers)),
        "n_sellers_eligible": int(len(result.clustered)),
        "n_sellers_excluded": int(len(result.new_low_data)),
        "multi_seller_order_stats": multi_seller_stats,
        "threshold_sensitivity": sensitivity.to_dict(orient="records"),
        "correlation_matrix": corr_matrix.to_dict(),
        "elbow_silhouette": elbow_df.to_dict(orient="records"),
        "chosen_k_silhouette": float(elbow_df.loc[elbow_df["k"] == config.N_CLUSTERS, "silhouette"].iloc[0]),
        "stability_ari": stability_df.to_dict(orient="records"),
        "stability_ari_mean": float(stability_df["ari"].mean()),
        "n_stability_seeds": len(config.STABILITY_SEEDS),
        "dbscan": dbscan_result,
        "dbscan_eps_sweep": dbscan_sweep,
        "dbscan_k_distance": kth_distances.tolist(),
        "pca_explained_variance_ratio": explained_var.tolist(),
        "pca_explained_variance_sum": float(explained_var.sum()),
        "health_score_weights": config.HEALTH_SCORE_WEIGHTS,
        "health_score_bounds": result.health_bounds,
    }
    with open(config.PROCESSED_FILES["model_diagnostics"], "w") as f:
        json.dump(diagnostics, f, indent=2)
    log("model_diagnostics.json written")

    log("Pipeline complete.")


if __name__ == "__main__":
    main()
