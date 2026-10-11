import numpy as np
import pytest

from src import clustering, config, features


@pytest.fixture(scope="module")
def X(seller_features) -> np.ndarray:
    eligible, _ = features.apply_min_order_threshold(seller_features)
    return eligible[config.CLUSTER_FEATURES].to_numpy()


def test_kmeans_is_deterministic_for_a_fixed_seed(X):
    a = clustering.fit_final_model(X, k=5, seed=config.SEED).named_steps["kmeans"].labels_
    b = clustering.fit_final_model(X, k=5, seed=config.SEED).named_steps["kmeans"].labels_
    np.testing.assert_array_equal(a, b)


def test_kmeans_produces_k_non_empty_clusters(X):
    labels = clustering.fit_final_model(X, k=5).named_steps["kmeans"].labels_
    assert sorted(set(labels)) == [0, 1, 2, 3, 4]


def test_ari_is_one_for_identical_partitions():
    labels = np.array([0, 0, 1, 1, 2, 2, 3])
    assert clustering.partition_agreement(labels, labels) == 1.0


def test_ari_ignores_label_permutation():
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([2, 2, 0, 0, 1, 1])
    assert clustering.partition_agreement(a, b) == 1.0


def test_ari_is_below_one_for_different_partitions():
    a = np.array([0, 0, 0, 1, 1, 1])
    b = np.array([0, 1, 0, 1, 0, 1])
    assert clustering.partition_agreement(a, b) < 1.0


def test_stability_ari_reports_every_seed_pair(X):
    seeds = [1, 2, 3]
    df = clustering.stability_ari(X, k=5, seeds=seeds)
    assert len(df) == 3
    assert df["ari"].between(-1, 1).all()


def test_stability_ari_is_one_when_seeds_repeat(X):
    df = clustering.stability_ari(X, k=5, seeds=[config.SEED, config.SEED])
    assert df["ari"].iloc[0] == 1.0
