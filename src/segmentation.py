"""End-to-end segmentation: seller features -> every seller with a named segment.

This module owns the rules that turn the clustering output into the business
view: the minimum-order split into a clustered population and the
"New / Low Data" bucket, health scoring with bounds fit on the clustered
population only, rank-based segment naming, and the per-segment records the
app reads from cluster_profiles.json.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.pipeline import Pipeline

from . import clustering, config, features, profiling


@dataclass
class SegmentationResult:
    sellers: pd.DataFrame
    """Every seller (clustered first, then New / Low Data) with cluster, health and segment columns."""
    profile_table: pd.DataFrame
    """Per-cluster medians and shares, computed against the full seller base."""
    cluster_to_key: dict[int, str]
    impacts: dict[int, str]
    health_bounds: dict
    model: Pipeline

    @property
    def clustered(self) -> pd.DataFrame:
        return self.sellers[self.sellers["cluster"] != config.NEW_LOW_DATA_CLUSTER]

    @property
    def new_low_data(self) -> pd.DataFrame:
        return self.sellers[self.sellers["cluster"] == config.NEW_LOW_DATA_CLUSTER]


def segment_sellers(
    seller_features: pd.DataFrame,
    k: int = config.N_CLUSTERS,
    seed: int = config.SEED,
    min_orders: int = config.MIN_ORDERS_THRESHOLD,
) -> SegmentationResult:
    """Cluster the eligible sellers and attach a segment to every seller.

    Sellers below `min_orders` never reach K-Means (their rate features are
    degenerate), but they are kept: they get cluster id -1, the
    "New / Low Data" segment, and a health score on the same scale as
    everyone else.
    """
    eligible, excluded = features.apply_min_order_threshold(seller_features, min_orders)

    model = clustering.fit_final_model(eligible[config.CLUSTER_FEATURES].to_numpy(), k=k, seed=seed)
    eligible = eligible.copy()
    eligible["cluster"] = model.named_steps["kmeans"].labels_

    # Bounds come from the eligible population only, so a handful of extreme
    # low-data sellers cannot compress the scale everyone is scored on.
    bounds = profiling.fit_health_score_bounds(eligible)
    eligible["health_score"] = profiling.compute_health_score(eligible, bounds)
    excluded = excluded.copy()
    excluded["health_score"] = profiling.compute_health_score(excluded, bounds)
    excluded["cluster"] = config.NEW_LOW_DATA_CLUSTER

    # Shares use the full seller base as denominator so the clustered
    # segments and the New / Low Data bucket sum to 100% together.
    profile_table = profiling.build_cluster_profile_table(
        eligible, total_revenue=seller_features["total_revenue"].sum(), total_sellers=len(seller_features)
    )
    cluster_to_key = profiling.name_segments(profile_table, eligible)
    impacts = profiling.quantify_recommendations(profile_table, cluster_to_key)

    eligible["segment_key"] = eligible["cluster"].map(cluster_to_key)
    eligible["segment_name"] = eligible["segment_key"].map(lambda key: profiling.SEGMENT_META[key]["name"])
    excluded["segment_key"] = config.NEW_LOW_DATA_KEY
    excluded["segment_name"] = config.NEW_LOW_DATA_NAME

    sellers = pd.concat([eligible, excluded], ignore_index=True, sort=False)
    sellers["segment_color"] = sellers["segment_key"].map(
        lambda key: config.SEGMENT_COLORS.get(key, config.EXCLUDED_SEGMENT_COLOR)
    )
    sellers["health_tier"] = sellers["health_score"].apply(profiling.health_tier)

    return SegmentationResult(
        sellers=sellers,
        profile_table=profile_table,
        cluster_to_key=cluster_to_key,
        impacts=impacts,
        health_bounds=bounds,
        model=model,
    )


def build_profile_records(result: SegmentationResult, min_orders: int = config.MIN_ORDERS_THRESHOLD) -> list[dict]:
    """One record per segment, as serialized to cluster_profiles.json."""
    records = []
    for _, row in result.profile_table.iterrows():
        cid = int(row["cluster"])
        key = result.cluster_to_key[cid]
        meta = profiling.SEGMENT_META[key]
        records.append(
            {
                "cluster": cid,
                "segment_key": key,
                "name": meta["name"],
                "action_tag": meta["action_tag"],
                "description": meta["description"],
                "action": meta["action"],
                "impact": result.impacts.get(cid, ""),
                "color": config.SEGMENT_COLORS[key],
                "n_sellers": int(row["n_sellers"]),
                "seller_share_pct": round(float(row["seller_share_pct"]), 2),
                "revenue_share_pct": round(float(row["revenue_share_pct"]), 2),
                "median_revenue": round(float(row["median_revenue"]), 2),
                "median_frequency": round(float(row["median_frequency"]), 2),
                "median_recency_days": round(float(row["median_recency_days"]), 1),
                "median_avg_delay_days": round(float(row["median_avg_delay_days"]), 2),
                "median_neg_review_rate": round(float(row["median_neg_review_rate"]), 4),
                "median_cancel_rate": round(float(row["median_cancel_rate"]), 4),
                "median_health": round(float(row["median_health"]), 1),
            }
        )
    records.append(_new_low_data_record(result.new_low_data, result.sellers, min_orders))
    return records


def _new_low_data_record(excluded: pd.DataFrame, all_sellers: pd.DataFrame, min_orders: int) -> dict:
    excluded_revenue_share = 100 * excluded["total_revenue"].sum() / all_sellers["total_revenue"].sum()
    excluded_seller_share = 100 * len(excluded) / len(all_sellers)

    def median(col: str, ndigits: int) -> float:
        return round(float(excluded[col].median()), ndigits) if len(excluded) else 0

    return {
        "cluster": config.NEW_LOW_DATA_CLUSTER,
        "segment_key": config.NEW_LOW_DATA_KEY,
        "name": config.NEW_LOW_DATA_NAME,
        "action_tag": "MONITOR",
        "description": (
            f"Sellers with fewer than {min_orders} distinct orders. Rate-based features "
            "(review, cancellation) are degenerate at this volume, so they are excluded from the K-Means "
            "population rather than distorting cluster centroids -- but they are never dropped from the "
            "business view."
        ),
        "action": (
            "No formal segmentation yet. Monitor early quality signals (first-order review, first delivery "
            f"delay) and re-evaluate once a seller crosses {min_orders} orders."
        ),
        "impact": (
            f"{len(excluded):,} sellers ({excluded_seller_share:.1f}% of the base) contribute "
            f"{excluded_revenue_share:.1f}% of total GMV -- excluding them from the "
            "clustering model costs negligible revenue coverage."
        ),
        "color": config.EXCLUDED_SEGMENT_COLOR,
        "n_sellers": int(len(excluded)),
        "seller_share_pct": round(excluded_seller_share, 2),
        "revenue_share_pct": round(excluded_revenue_share, 2),
        "median_revenue": median("total_revenue", 2),
        "median_frequency": median("frequency", 2),
        "median_recency_days": median("recency_days", 1),
        "median_avg_delay_days": median("avg_delay_days", 2),
        "median_neg_review_rate": median("neg_review_rate", 4),
        "median_cancel_rate": median("cancel_rate", 4),
        "median_health": median("health_score", 1),
    }
