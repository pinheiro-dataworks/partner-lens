<p align="center">
  <img src="app/assets/Logo_White.png" alt="PartnerLens" width="220">
</p>

# PartnerLens

**Partner segmentation & performance intelligence for a marketplace, built on the Olist Brazilian E-Commerce dataset.**

PartnerLens segments Olist **sellers** — not customers, not orders — into named, business-actionable partner profiles, scores each one on a 0–100 health composite, and turns each segment into a quantified recommendation. Everything in the app is computed from the real dataset; there are no placeholder numbers.

> Developed by [Renan Pinheiro](https://github.com/pinheiro-dataworks)

[![CI](https://github.com/pinheiro-dataworks/partner-lens/actions/workflows/ci.yml/badge.svg)](https://github.com/pinheiro-dataworks/partner-lens/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

![Overview tab](docs/screenshots/overview.png)

## Problem

Marketplaces live or die on the quality of their partner base, but a platform with thousands of sellers can't manage them one at a time. Olist has 3,095 sellers with wildly different scale, activity and quality — treating them as a single undifferentiated group means over-investing in partners who don't need it and under-reacting to the ones quietly damaging buyer trust. PartnerLens exists to turn that undifferentiated list into a small number of segments a partnerships team can actually act on.

## Approach

- **Unit of analysis: the seller-order pair, not the raw order.** Olist is a marketplace — a single `order_id` in `order_items` can contain items from more than one seller. Aggregating straight from `order_items` inflates a seller's order frequency; aggregating straight from `orders` credits a seller with revenue from items they didn't sell. Every feature is built by first materializing the seller × order_id grain, then rolling up to seller level. See [`src/features.py`](src/features.py).
- **Known limitation, measured rather than assumed.** Reviews attach to the order, not the line item, so a multi-seller order assigns the same review score to every seller involved. Measured directly from the data: **1.3%** of orders (**1.98%** of revenue) are multi-seller — the distortion is real but marginal.
- **A minimum-order threshold, with the sensitivity analysis to defend it.** Sellers with too few orders have degenerate rate features (a negative-review rate can only land on 0%, 50% or 100% at 1–2 orders). At the chosen threshold (≥5 orders), the model covers **58.0%** of sellers and **95.8%** of revenue. The excluded 1,301 sellers aren't dropped — they're the "New / Low Data" segment throughout the app.
- **`log1p` on monetary/count features before scaling**, because revenue and order volume are heavily right-skewed by a handful of high-GMV sellers.
- **A correlation check that changed the feature set.** `log_revenue` and `log_frequency` correlate at 0.80 on the eligible population — past the ~0.7 flag threshold. Frequency was dropped from the K-Means feature set (revenue already captures scale) but kept in the health score and every profile card.
- **K-Means (k=5), chosen on business interpretability, not silhouette alone.** Silhouette's global maximum is the coarser k=2 solution (0.405); k=5 is a local peak (0.250) rather than the statistical optimum — expected for continuous behavioral data with no natural density valleys. k=5 wins because coarser k values collapse segments that need opposite actions into one bucket.
- **A stability check most portfolios skip.** K-Means refit across 5 seeds and compared with the Adjusted Rand Index gives a mean pairwise ARI of **0.980** — the partition is essentially deterministic, not an artifact of initialization.
- **DBSCAN as a critical stress test, not a second candidate.** Density-based clustering doesn't fit this feature space by construction (there are no natural density valleys between "good" and "bad" sellers); the eps sweep confirms it — noise falls monotonically as eps grows but cluster count never exceeds 1. The noise set is reframed as an outlier-detection cross-check on K-Means instead of a discarded experiment.
- **PCA for the scatter plot only, never the clustering space.** PC1+PC2 explain 53.6% of variance — below the ~60% mark needed to trust it as a distance map, so the app labels the projection illustrative. Clustering runs on the original scaled features so centroids stay interpretable in business units.
- **The app trains nothing at runtime.** `scripts/build_pipeline.py` is the offline half; `app/streamlit_app.py` only reads pre-computed Parquet/JSON artifacts from `data/processed/`, which keeps boot fast on Streamlit Community Cloud and keeps "how was this trained" separate from "how is this served."

Full reasoning, with every chart the decisions above are based on, lives in the app's **Methodology** tab and in `notebooks/`.

## Segments

| Segment | Sellers | % of base | % of GMV | Median health | Action |
|---|---:|---:|---:|---:|---|
| Premium & Consistent | 678 | 21.9% | 80.8% | 73 | RETAIN |
| Emerging | 608 | 19.6% | 5.6% | 64 | GROW |
| Underperforming – Quality Risk | 254 | 8.2% | 5.2% | 57 | DEVELOP |
| Declining | 201 | 6.5% | 3.4% | 57 | REACTIVATE |
| New / Low Data (< 5 orders) | 1,301 | 42.0% | 4.2% | 53 | MONITOR |
| At Risk / Low Engagement | 53 | 1.7% | 0.8% | 51 | DEPRIORITIZE |

Names and descriptions are assigned from where each cluster **ranks** against the others on health, recency and quality — never from a fixed cluster id, which is arbitrary and not stable across re-runs. Worth flagging: the original design sketch for this project anticipated a "high volume, low quality" archetype. It doesn't appear in the real k=5 solution — above-median order frequency only shows up in the healthiest cluster. Quality problems concentrate in **low/mid-volume** sellers instead, so the segment names reflect what the data actually shows rather than a preconceived narrative.

![Segment Explorer tab](docs/screenshots/segment_explorer.png)

## Actions & impact

Every profile card carries a quantified impact statement computed from the actual segmented data, for example:

- **Premium & Consistent**: 21.9% of sellers generate 80.8% of GMV — a 3.7x revenue concentration relative to headcount, the strongest ratio of any segment.
- **Underperforming – Quality Risk**: runs a 28.6% negative-review rate against a ~13% median elsewhere, while holding only 5.2% of GMV — a remediation plan is cheaper here than a revenue write-off.
- **New / Low Data**: 42.0% of the seller base but only 4.2% of GMV — excluding this tail from the clustering model costs negligible revenue coverage.

See the **Segment Explorer** tab for the full set, each with a specific recommended action and the health-score distribution *within* the segment (a "healthy" segment can still carry a lower tail worth a manual look).

![Seller Lookup tab](docs/screenshots/seller_lookup.png)

## Reproducing the reported numbers

Every figure in this README comes from the versioned artifacts in `data/processed/`, which a fresh run of `python scripts/build_pipeline.py` reproduces. The headline figures and where to find them:

| Claim | Value | Source |
|---|---:|---|
| Premium & Consistent share of sellers | 21.9% | `cluster_profiles.json` → `premium.seller_share_pct` (21.91) |
| Premium & Consistent share of GMV | 80.8% | `cluster_profiles.json` → `premium.revenue_share_pct` (80.80) |
| Mean pairwise ARI, k=5, 5 seeds | 0.980 | `model_diagnostics.json` → `stability_ari_mean` (0.9801) |
| Silhouette at k=5 / k=2 | 0.250 / 0.405 | `model_diagnostics.json` → `elbow_silhouette` |
| Sellers / revenue kept at ≥5 orders | 58.0% / 95.8% | `model_diagnostics.json` → `threshold_sensitivity` |
| Multi-seller orders / revenue | 1.3% / 1.98% | `model_diagnostics.json` → `multi_seller_order_stats` |
| PCA variance explained (PC1+PC2) | 53.6% | `model_diagnostics.json` → `pca_explained_variance_sum` |

K-Means runs with a fixed seed, so the segmentation itself is deterministic. Across machines, only the last floating-point digits of the K-Means inertia values in `model_diagnostics.json` change, and no reported figure moves.

## Limitations

- **Review attribution in multi-seller orders.** Olist attaches reviews to the order, not the line item, so every seller in a multi-seller order inherits the same score. This affects 1.3% of orders (1.98% of revenue). The source schema causes it, and no downstream modeling choice can fix it.
- **A snapshot, not a trend.** The dataset ends on 2018-10-17, and recency is measured against that date. The segmentation describes sellers at one point in time and says nothing about how they move between segments.
- **Health-score weights are a business judgment.** The 0.55 quality / 0.45 volume split encodes a stated priority. It is not fitted to, or validated against, an outcome such as seller churn or GMV loss, and the score's sensitivity to the weights has not been analyzed.
- **Modest cluster separation.** The silhouette at k=5 is 0.250. Seller behavior is a continuum without density gaps, as the DBSCAN stress test confirms, so segment boundaries are useful cut points rather than natural groups. The PCA scatter covers 53.6% of the variance and is illustrative only.
- **Segment naming assumes k=5.** `profiling.name_segments` assigns five named profiles by rank, and the descriptive copy in `profiling.SEGMENT_META` was written against this run's medians. If you change `n_clusters`, the threshold or the feature set in `configs/pipeline.yaml`, re-read that copy before publishing.
- **Tests run on synthetic data.** The pytest suite checks the logic and its invariants on a small generated fixture. Matching the real-data numbers requires the Kaggle CSVs and a pipeline run, which CI does not do because the raw data is not redistributed in this repository.

## Next steps

- Track segment membership over time (rolling reference dates) so that transitions such as Emerging → Premium can be measured instead of assumed.
- Surface the DBSCAN noise set as a manual-review worklist in the app. It is computed and stored in `sellers_segmented.parquet` but not yet shown.
- Run a sensitivity analysis on the health-score weights.

## Repository structure

```
partner-lens/
├── .github/workflows/ci.yml   # ruff + pytest on every pull request and push to main
├── app/
│   ├── streamlit_app.py       # reads data/processed/ only, trains nothing
│   └── assets/Logo_White.png
├── configs/
│   └── pipeline.yaml          # seed, thresholds, k, health-score weights and tiers
├── data/
│   ├── raw/                   # Olist CSVs from Kaggle (gitignored)
│   └── processed/             # artifacts served by the app (see Data policy)
├── design/
│   └── partnerlens_dashboard_prototype.html   # original static design sketch
├── notebooks/                 # 01_eda -> 02_features -> 03_clustering, executed
├── scripts/
│   ├── build_pipeline.py      # raw CSVs -> data/processed/ (diagnostics + I/O)
│   └── build_notebooks.py     # (re)generates notebooks/ from cell definitions
├── src/
│   ├── config.py              # loads configs/pipeline.yaml; paths, labels, palette
│   ├── data_loader.py         # raw CSV loading + referential-integrity checks
│   ├── features.py            # seller-order pairs -> seller-level feature matrix
│   ├── clustering.py          # K-Means pipeline, k-selection, ARI stability, DBSCAN, PCA
│   ├── profiling.py           # health score, segment naming, quantified recommendations
│   └── segmentation.py        # minimum-order split, New / Low Data rule, segment assignment
├── tests/                     # pytest suite on a synthetic Olist-shaped fixture
├── pyproject.toml             # ruff + pytest configuration
├── requirements.txt           # pinned runtime dependencies (pipeline + app)
├── requirements-dev.txt       # + notebooks, pytest, ruff
└── LICENSE
```

## How to run

Requires Python 3.11 or newer (CI runs 3.12). Run every command from the repository root.

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# 2. Data: the 9 Olist CSVs into data/raw/
#    Needs a Kaggle API token (https://www.kaggle.com/docs/api). You can also
#    download the zip manually from
#    https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce and extract it to data/raw/.
pip install kaggle
kaggle datasets download olistbr/brazilian-ecommerce -p data/raw --unzip

# 3. Features + training: validates the raw tables, builds seller-order pairs and
#    seller features, runs the k-selection / stability / DBSCAN diagnostics, fits
#    K-Means and writes data/processed/
python scripts/build_pipeline.py

# 4. Notebooks: regenerate from scripts/build_notebooks.py, then execute
python scripts/build_notebooks.py
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_features.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_clustering.ipynb

# 5. App
streamlit run app/streamlit_app.py

# 6. Tests and lint (no raw data needed)
pytest
ruff check .
```

The app and the test suite need only the versioned artifacts, so you can skip steps 2–4 to browse the app or run the tests.

## Data policy

- **`data/raw/`** is not versioned. The Olist dataset is published on Kaggle under its own license (CC BY-NC-SA 4.0), so download it from the source instead of copying it here.
- **`data/processed/`** versions only what the app serves: `sellers_segmented.parquet`, `cluster_profiles.json`, `model_diagnostics.json` and `data_quality_report.json`. This lets the app and the test suite run on a fresh clone without the raw data.
- **Pipeline intermediates are not versioned.** `seller_order_pairs.parquet` (~6 MB) and `seller_features_all.parquet` are written by `scripts/build_pipeline.py` for inspection and debugging. Neither the app nor the notebooks read them, and step 3 above regenerates them.
- **Regenerate artifacts whenever `src/` or `configs/` changes.** Commit the regenerated served artifacts in the same pull request as the code change, so the numbers the app shows always match the code.

## Notebooks

- `01_eda.ipynb` covers relational integrity checks, the multi-seller-order limitation quantified, and the revenue and review-score distributions.
- `02_features.ipynb` covers seller-order pair construction, the minimum-order threshold sensitivity analysis and the log1p skew treatment.
- `03_clustering.ipynb` covers feature correlation and the decision to drop frequency, k-selection (elbow, silhouette and stability), the DBSCAN stress test, PCA caveats, and final segment naming.

The notebooks import the same `src/` functions as the pipeline and the app. They narrate the decisions but do not re-implement the logic.

## Stack

Python, pandas, scikit-learn (StandardScaler + KMeans pipeline, DBSCAN, PCA), Plotly, Streamlit, pytest, ruff, GitHub Actions. No model is trained at app runtime.

## Development history

Developed locally and published as a snapshot; subsequent changes tracked via pull requests.

## License

[MIT](LICENSE). The Olist dataset is not covered by this license; see its [Kaggle page](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) for its terms.
