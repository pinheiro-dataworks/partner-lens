"""Shared fixtures: a small, synthetic Olist-shaped dataset.

The tables mirror the columns the pipeline reads from the real CSVs, at a
size that runs in well under a second. Edge cases the real data contains are
planted deliberately so the tests exercise them:

- sellers with exactly 4 and exactly 5 orders (both sides of the threshold);
- a seller whose only orders were canceled (no delivery signal to measure);
- a seller with no reviews at all (no review signal to measure);
- an order shared by two sellers (multi-seller attribution);
- an order with two review rows (dedupe keeps the most recent).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import features

REFERENCE_START = pd.Timestamp("2017-01-01")

# seller_id -> number of distinct orders. Ten sellers below the threshold of
# 5, forty at or above it so K-Means has enough points for k=5.
ORDER_COUNTS = {
    **{f"low_{i:02d}": n for i, n in enumerate([1, 1, 2, 2, 3, 3, 4, 4, 4, 1])},
    "edge_exactly_5": 5,
    **{f"seller_{i:02d}": n for i, n in enumerate(np.random.default_rng(7).integers(5, 60, size=39))},
}
CANCELED_ONLY_SELLER = "low_09"
NO_REVIEW_SELLER = "seller_00"


def _build_tables(seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    orders, items, reviews = [], [], []

    order_no = 0
    for seller_id, n_orders in ORDER_COUNTS.items():
        # Per-seller tendencies give K-Means some real structure to find.
        delay_bias = rng.normal(0, 6)
        bad_review_p = rng.uniform(0.02, 0.5)
        cancel_p = rng.uniform(0, 0.15)
        ticket = rng.lognormal(4.5, 0.8)
        last_active = rng.uniform(0, 600)

        for _ in range(n_orders):
            order_id = f"o{order_no:05d}"
            order_no += 1
            purchased = REFERENCE_START + pd.Timedelta(days=float(rng.uniform(0, 600 - last_active / 2)))
            canceled = seller_id == CANCELED_ONLY_SELLER or rng.random() < cancel_p
            estimated = purchased + pd.Timedelta(days=20)
            delivered = None if canceled else estimated + pd.Timedelta(days=float(delay_bias + rng.normal(0, 3)))
            orders.append(
                {
                    "order_id": order_id,
                    "order_status": "canceled" if canceled else "delivered",
                    "order_purchase_timestamp": purchased,
                    "order_delivered_customer_date": delivered,
                    "order_estimated_delivery_date": estimated,
                }
            )
            for item_no in range(1, int(rng.integers(1, 3)) + 1):
                items.append(
                    {
                        "order_id": order_id,
                        "order_item_id": item_no,
                        "seller_id": seller_id,
                        "price": round(float(ticket * rng.uniform(0.7, 1.3)), 2),
                        "freight_value": round(float(rng.uniform(5, 30)), 2),
                    }
                )
            if seller_id != NO_REVIEW_SELLER:
                score = 1 if rng.random() < bad_review_p else int(rng.integers(3, 6))
                reviews.append(
                    {
                        "order_id": order_id,
                        "review_score": score,
                        "review_answer_timestamp": purchased + pd.Timedelta(days=25),
                    }
                )

    orders_df = pd.DataFrame(orders)
    items_df = pd.DataFrame(items)
    reviews_df = pd.DataFrame(reviews)

    # Multi-seller order: seller_02 also ships an item in seller_01's first order.
    shared_order = items_df.loc[items_df["seller_id"] == "seller_01", "order_id"].iloc[0]
    items_df = pd.concat(
        [
            items_df,
            pd.DataFrame(
                [{"order_id": shared_order, "order_item_id": 9, "seller_id": "seller_02", "price": 50.0, "freight_value": 10.0}]
            ),
        ],
        ignore_index=True,
    )

    # Duplicate review row for one order; the later answer must win.
    first = reviews_df.iloc[0]
    reviews_df = pd.concat(
        [
            reviews_df,
            pd.DataFrame(
                [
                    {
                        "order_id": first["order_id"],
                        "review_score": 5 if first["review_score"] <= 2 else 1,
                        "review_answer_timestamp": first["review_answer_timestamp"] + pd.Timedelta(days=3),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    sellers_df = pd.DataFrame(
        {
            "seller_id": list(ORDER_COUNTS),
            "seller_state": "SP",
            "seller_city": "sao paulo",
        }
    )
    return {"orders": orders_df, "order_items": items_df, "order_reviews": reviews_df, "sellers": sellers_df}


@pytest.fixture(scope="session")
def raw_tables() -> dict[str, pd.DataFrame]:
    return _build_tables()


@pytest.fixture(scope="session")
def feature_table(raw_tables):
    """(pairs, seller_features, reference_date) built the same way the pipeline builds them."""
    return features.build_feature_table(raw_tables)


@pytest.fixture(scope="session")
def seller_features(feature_table) -> pd.DataFrame:
    return feature_table[1]
