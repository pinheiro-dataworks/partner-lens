import pytest

from src import config, profiling, segmentation


@pytest.fixture(scope="module")
def result(seller_features) -> segmentation.SegmentationResult:
    return segmentation.segment_sellers(seller_features, k=5, seed=config.SEED, min_orders=5)


def test_config_threshold_is_five_orders():
    assert config.MIN_ORDERS_THRESHOLD == 5


def test_sellers_below_threshold_are_new_low_data(result):
    sellers = result.sellers
    low = sellers[sellers["frequency"] < 5]
    assert len(low) > 0
    assert (low["cluster"] == config.NEW_LOW_DATA_CLUSTER).all()
    assert (low["segment_key"] == config.NEW_LOW_DATA_KEY).all()
    assert (low["segment_name"] == config.NEW_LOW_DATA_NAME).all()


def test_sellers_at_or_above_threshold_are_clustered(result):
    sellers = result.sellers
    eligible = sellers[sellers["frequency"] >= 5]
    assert (eligible["cluster"] != config.NEW_LOW_DATA_CLUSTER).all()
    assert config.NEW_LOW_DATA_KEY not in set(eligible["segment_key"])


@pytest.mark.parametrize(("seller_id", "expected_key_is_low_data"), [("low_06", True), ("edge_exactly_5", False)])
def test_threshold_boundary(result, seller_id, expected_key_is_low_data):
    row = result.sellers.set_index("seller_id").loc[seller_id]
    assert (row["segment_key"] == config.NEW_LOW_DATA_KEY) is expected_key_is_low_data


def test_every_seller_gets_exactly_one_segment(result, seller_features):
    assert len(result.sellers) == len(seller_features)
    assert result.sellers["seller_id"].is_unique
    assert result.sellers["segment_key"].notna().all()


def test_each_cluster_gets_a_distinct_business_name(result):
    assert sorted(result.cluster_to_key.values()) == sorted(profiling.SEGMENT_META)
    assert len(set(result.cluster_to_key)) == 5


def test_profile_records_cover_all_sellers_and_gmv(result):
    records = segmentation.build_profile_records(result, min_orders=5)
    assert sum(r["n_sellers"] for r in records) == len(result.sellers)
    assert sum(r["revenue_share_pct"] for r in records) == pytest.approx(100, abs=0.05)
    low = next(r for r in records if r["segment_key"] == config.NEW_LOW_DATA_KEY)
    assert low["n_sellers"] == len(result.new_low_data)


def test_segmentation_is_deterministic_for_a_fixed_seed(seller_features, result):
    again = segmentation.segment_sellers(seller_features, k=5, seed=config.SEED, min_orders=5)
    assert (again.sellers["cluster"].to_numpy() == result.sellers["cluster"].to_numpy()).all()
    assert again.cluster_to_key == result.cluster_to_key
