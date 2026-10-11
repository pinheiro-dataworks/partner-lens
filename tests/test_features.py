import pandas as pd

from src import config, data_loader, features
from tests.conftest import CANCELED_ONLY_SELLER, NO_REVIEW_SELLER, ORDER_COUNTS

# Columns the clustering model, health score or app read directly. None of
# them may be null: missing delivery/review signal is imputed and flagged.
MODEL_COLUMNS = sorted(
    set(config.RAW_FEATURES)
    | set(config.CANDIDATE_FEATURES)
    | set(config.CLUSTER_FEATURES)
    | {"avg_delivery_days", "avg_ticket", "n_delivered_orders", "last_order_date"}
)


def test_one_row_per_seller(seller_features):
    assert seller_features["seller_id"].is_unique
    assert set(seller_features["seller_id"]) == set(ORDER_COUNTS)


def test_model_features_have_no_nulls(seller_features):
    nulls = seller_features[MODEL_COLUMNS].isna().sum()
    assert nulls.sum() == 0, nulls[nulls > 0].to_dict()


def test_missing_signals_are_imputed_and_flagged(seller_features):
    by_id = seller_features.set_index("seller_id")
    assert by_id.loc[CANCELED_ONLY_SELLER, "delivery_data_imputed"]
    assert by_id.loc[CANCELED_ONLY_SELLER, "n_delivered_orders"] == 0
    assert by_id.loc[NO_REVIEW_SELLER, "review_data_imputed"]
    # The flag marks exactly the sellers with no measured signal, nobody else.
    assert (seller_features["delivery_data_imputed"] == (seller_features["n_delivered_orders"] == 0)).all()
    assert seller_features["review_data_imputed"].sum() == 1


def test_rates_are_proportions(seller_features):
    for col in ("cancel_rate", "neg_review_rate"):
        assert seller_features[col].between(0, 1).all(), col
    assert seller_features.set_index("seller_id").loc[CANCELED_ONLY_SELLER, "cancel_rate"] == 1.0


def test_frequency_counts_distinct_orders_not_items(raw_tables, seller_features):
    # Orders in the fixture carry 1-2 items each; frequency must not count items.
    expected = raw_tables["order_items"].groupby("seller_id")["order_id"].nunique()
    got = seller_features.set_index("seller_id")["frequency"]
    pd.testing.assert_series_equal(got.sort_index(), expected.sort_index(), check_names=False)


def test_multi_seller_order_splits_revenue_by_seller(raw_tables, feature_table):
    pairs = feature_table[0]
    items = raw_tables["order_items"]
    shared = items.groupby("order_id")["seller_id"].nunique().loc[lambda s: s > 1].index
    assert len(shared) == 1
    shared_pairs = pairs[pairs["order_id"] == shared[0]].set_index("seller_id")["seller_revenue"]
    expected = items[items["order_id"] == shared[0]].groupby("seller_id")["price"].sum()
    pd.testing.assert_series_equal(shared_pairs.sort_index(), expected.sort_index(), check_names=False)


def test_dedupe_reviews_keeps_latest_answer(raw_tables):
    reviews = raw_tables["order_reviews"]
    dup_order = reviews.loc[reviews["order_id"].duplicated(), "order_id"].iloc[0]
    latest = reviews[reviews["order_id"] == dup_order].sort_values("review_answer_timestamp").iloc[-1]

    deduped = data_loader.dedupe_reviews(reviews)
    assert deduped["order_id"].is_unique
    assert deduped.set_index("order_id").loc[dup_order, "review_score"] == latest["review_score"]


def test_recency_is_measured_against_dataset_reference_date(raw_tables, feature_table):
    _, seller_features, reference_date = feature_table
    assert reference_date == raw_tables["orders"]["order_purchase_timestamp"].max()
    assert (seller_features["recency_days"] >= 0).all()
    assert seller_features["recency_days"].min() == 0


def test_min_order_threshold_split_is_exhaustive(seller_features):
    eligible, excluded = features.apply_min_order_threshold(seller_features, min_orders=5)
    assert len(eligible) + len(excluded) == len(seller_features)
    assert (eligible["frequency"] >= 5).all()
    assert (excluded["frequency"] < 5).all()
