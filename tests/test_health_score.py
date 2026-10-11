import numpy as np
import pandas as pd
import pytest

from src import config, features, profiling

# Round-number bounds so each component can be driven to exactly 0 or 1.
BOUNDS = {
    "total_revenue": (0.0, 10.0),  # on the log1p scale
    "frequency": (0.0, 5.0),  # on the log1p scale
    "recency_days": (0.0, 100.0),
    "avg_delay_days": (-10.0, 10.0),
}
BEST_VOLUME = {"total_revenue": np.expm1(10.0), "frequency": np.expm1(5.0), "recency_days": 0.0}
WORST_VOLUME = {"total_revenue": 0.0, "frequency": 0.0, "recency_days": 100.0}
BEST_QUALITY = {"avg_delay_days": -10.0, "neg_review_rate": 0.0, "cancel_rate": 0.0}
WORST_QUALITY = {"avg_delay_days": 10.0, "neg_review_rate": 1.0, "cancel_rate": 1.0}


def _score(**row) -> float:
    return float(profiling.compute_health_score(pd.DataFrame([row]), BOUNDS).iloc[0])


def test_weights_sum_to_one():
    assert sum(config.HEALTH_SCORE_WEIGHTS.values()) == pytest.approx(1.0)


def test_weight_groups_partition_the_components():
    grouped = config.HEALTH_SCORE_GROUPS["quality"] + config.HEALTH_SCORE_GROUPS["volume"]
    assert sorted(grouped) == sorted(config.HEALTH_SCORE_WEIGHTS)


def test_config_weights_quality_over_volume():
    w = config.HEALTH_SCORE_WEIGHTS
    assert sum(w[k] for k in config.HEALTH_SCORE_GROUPS["quality"]) == pytest.approx(0.55)
    assert sum(w[k] for k in config.HEALTH_SCORE_GROUPS["volume"]) == pytest.approx(0.45)


def test_extremes_map_to_0_and_100():
    assert _score(**BEST_VOLUME, **BEST_QUALITY) == pytest.approx(100.0)
    assert _score(**WORST_VOLUME, **WORST_QUALITY) == pytest.approx(0.0)


def test_score_applies_the_quality_and_volume_weights():
    # Perfect quality with zero volume earns exactly the quality weight, and
    # vice versa: the 0.55 / 0.45 split from the config is what the score uses.
    assert _score(**WORST_VOLUME, **BEST_QUALITY) == pytest.approx(55.0)
    assert _score(**BEST_VOLUME, **WORST_QUALITY) == pytest.approx(45.0)


def test_score_uses_the_weights_it_is_given():
    row = pd.DataFrame([{**BEST_VOLUME, **WORST_QUALITY}])
    revenue_only = {k: (1.0 if k == "revenue" else 0.0) for k in config.HEALTH_SCORE_WEIGHTS}
    assert profiling.compute_health_score(row, BOUNDS, weights=revenue_only).iloc[0] == pytest.approx(100.0)


def test_score_is_clipped_for_values_outside_the_fitted_bounds():
    extreme = pd.DataFrame(
        [
            {"total_revenue": 1e12, "frequency": 1e9, "recency_days": -50, "avg_delay_days": -400, "neg_review_rate": -1, "cancel_rate": -1},
            {"total_revenue": 0, "frequency": 0, "recency_days": 1e6, "avg_delay_days": 400, "neg_review_rate": 3, "cancel_rate": 3},
        ]
    )
    scores = profiling.compute_health_score(extreme, BOUNDS)
    assert scores.between(0, 100).all()


def test_scores_on_fixture_are_within_0_100(seller_features):
    eligible, excluded = features.apply_min_order_threshold(seller_features)
    bounds = profiling.fit_health_score_bounds(eligible)
    for population in (eligible, excluded):
        scores = profiling.compute_health_score(population, bounds)
        assert scores.notna().all()
        assert scores.between(0, 100).all()


def test_components_are_oriented_so_one_is_healthier():
    good = profiling.health_components(pd.DataFrame([{**BEST_VOLUME, **BEST_QUALITY}]), BOUNDS).iloc[0]
    bad = profiling.health_components(pd.DataFrame([{**WORST_VOLUME, **WORST_QUALITY}]), BOUNDS).iloc[0]
    assert (good == 1.0).all()
    assert (bad == 0.0).all()
    assert list(good.index) == list(config.HEALTH_SCORE_WEIGHTS)


@pytest.mark.parametrize(
    ("score", "tier"),
    [(100, "Strong"), (75, "Strong"), (74.9, "Stable"), (50, "Stable"), (49.9, "Fragile"), (30, "Fragile"), (29.9, "Critical"), (0, "Critical")],
)
def test_health_tiers_follow_config_cutoffs(score, tier):
    assert profiling.health_tier(score) == tier
