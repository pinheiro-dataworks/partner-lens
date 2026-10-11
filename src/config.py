"""Central configuration for the PartnerLens pipeline.

Modeling parameters (seed, thresholds, k, health-score weights) are read from
configs/pipeline.yaml; repository paths, labels and the app palette are
defined here. Every other module imports from this one, so there is a single
place to answer "where is X decided?".
"""
from pathlib import Path

import yaml

# ---------------------------------------------------------------- paths ----
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
APP_ASSETS_DIR = ROOT_DIR / "app" / "assets"

RAW_FILES = {
    "orders": DATA_RAW_DIR / "olist_orders_dataset.csv",
    "order_items": DATA_RAW_DIR / "olist_order_items_dataset.csv",
    "order_payments": DATA_RAW_DIR / "olist_order_payments_dataset.csv",
    "order_reviews": DATA_RAW_DIR / "olist_order_reviews_dataset.csv",
    "products": DATA_RAW_DIR / "olist_products_dataset.csv",
    "sellers": DATA_RAW_DIR / "olist_sellers_dataset.csv",
    "customers": DATA_RAW_DIR / "olist_customers_dataset.csv",
    "geolocation": DATA_RAW_DIR / "olist_geolocation_dataset.csv",
    "category_translation": DATA_RAW_DIR / "product_category_name_translation.csv",
}

PROCESSED_FILES = {
    "seller_order_pairs": DATA_PROCESSED_DIR / "seller_order_pairs.parquet",
    "seller_features": DATA_PROCESSED_DIR / "seller_features_all.parquet",
    "sellers_segmented": DATA_PROCESSED_DIR / "sellers_segmented.parquet",
    "cluster_profiles": DATA_PROCESSED_DIR / "cluster_profiles.json",
    "model_diagnostics": DATA_PROCESSED_DIR / "model_diagnostics.json",
    "data_quality_report": DATA_PROCESSED_DIR / "data_quality_report.json",
}

# ------------------------------------------------------- modeling config --
# Seeds, thresholds, k and health-score weights are read from YAML so that a
# re-run with different settings is a config change rather than a code edit.
# The constants below are the stable import surface for src/, scripts,
# notebooks and the app.
CONFIG_PATH = ROOT_DIR / "configs" / "pipeline.yaml"


def load_pipeline_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


_cfg = load_pipeline_config()

SEED: int = _cfg["seed"]

MIN_ORDERS_THRESHOLD: int = _cfg["segmentation"]["min_orders_threshold"]

N_CLUSTERS: int = _cfg["clustering"]["n_clusters"]
K_CANDIDATES: list[int] = list(_cfg["clustering"]["k_candidates"])
KMEANS_N_INIT: int = _cfg["clustering"]["kmeans_n_init"]
STABILITY_SEEDS: list[int] = list(_cfg["clustering"]["stability_seeds"])

DBSCAN_MIN_SAMPLES: int = _cfg["dbscan"]["min_samples"]
DBSCAN_EPS_MULTIPLIERS: list[float] = list(_cfg["dbscan"]["eps_sweep_multipliers"])

# Features feeding the clustering model, in original business units before
# scaling. Monetary/count features are log1p-transformed (see features.py)
# because revenue and order volume are heavily right-skewed by a small number
# of high-GMV sellers, which would otherwise dominate Euclidean distance.
RAW_FEATURES = ["total_revenue", "frequency", "recency_days", "avg_delay_days", "neg_review_rate", "cancel_rate"]
CANDIDATE_FEATURES: list[str] = list(_cfg["features"]["candidate"])
CLUSTER_FEATURES: list[str] = list(_cfg["features"]["cluster"])

HEALTH_SCORE_WEIGHTS: dict[str, float] = dict(_cfg["health_score"]["weights"])
# Components grouped by what they measure. The quality/volume split is the
# business decision behind the weights (0.55 vs 0.45), so it is asserted in
# tests rather than left implicit in the numbers.
HEALTH_SCORE_GROUPS = {
    "quality": ["delay", "neg_review", "cancel"],
    "volume": ["revenue", "frequency", "recency"],
}
HEALTH_TIERS: dict[str, float] = dict(_cfg["health_score"]["tiers"])

ACTIVE_WINDOW_DAYS: int = _cfg["app"]["active_window_days"]

FEATURE_LABELS = {
    "log_revenue": "Revenue (log)",
    "log_frequency": "Frequency (log)",
    "recency_days": "Recency",
    "avg_delay_days": "Delivery delay",
    "neg_review_rate": "Neg. review rate",
    "cancel_rate": "Cancel rate",
    "total_revenue": "Revenue",
    "frequency": "Frequency",
}

# Direction correction so that +1 always means "healthier" once z-scored:
# recency, delay, negative reviews and cancellations are "bad when high", so
# they enter the health score inverted.
INVERTED_FEATURES = {"recency_days", "avg_delay_days", "neg_review_rate", "cancel_rate"}

# ------------------------------------------------------------------ design --
PALETTE = {
    "bg": "#14171C",
    "bg_elevated": "#1B1F27",
    "surface": "#1F242E",
    "surface_hover": "#262C38",
    "border": "#2B303C",
    "border_strong": "#3A4152",
    "text_primary": "#ECE9E2",
    "text_secondary": "#9CA1AD",
    "text_tertiary": "#666C79",
    "gold": "#E8A33D",
    "rust": "#C1694A",
    "violet": "#8B85C7",
    "teal": "#4FB3A9",
    "red": "#D6685F",
    "green": "#6FBF7D",
}

# Keyed by segment key, never by cluster id: raw K-Means labels are arbitrary
# and not stable in meaning across re-runs (see profiling.name_segments()).
SEGMENT_COLORS = {
    "premium": PALETTE["gold"],
    "emerging": PALETTE["teal"],
    "declining": PALETTE["violet"],
    "underperforming": PALETTE["rust"],
    "atrisk": PALETTE["red"],
}
EXCLUDED_SEGMENT_COLOR = PALETTE["text_tertiary"]

# Sellers below MIN_ORDERS_THRESHOLD are not clustered; they carry this
# sentinel cluster id and segment instead.
NEW_LOW_DATA_CLUSTER = -1
NEW_LOW_DATA_KEY = "new_low_data"
NEW_LOW_DATA_NAME = "New / Low Data"

PROJECT_NAME = "PartnerLens"
PROJECT_TAGLINE = "Partner Segmentation & Performance Intelligence"
AUTHOR_NAME = "Renan Pinheiro"
AUTHOR_GITHUB = "https://github.com/pinheiro-dataworks"
