# Grouping With ML — implementation and handoff

Updated 2026-11-10. The user confirms that the website deployed successfully.
The third main tab now has **K-means**, **Hierarchical clustering**, and
**Adding a direction**. All three use January 2005–December 2009 training
validation, after the initial history. Opening them never fits or replays.

## What the two methods learn

Each stock has a row of twelve already-earned monthly simple returns. Center
and scale each stock by its own mean and population standard deviation. Complete,
non-constant paths only. On these standardized paths, squared Euclidean distance
is proportional to one minus return correlation. Fit five groups each month;
IDs are temporary and are not industries or persistent cross-month labels.
Neither factor ranks nor forward-return labels are grouping inputs.

K-means alternates nearest-centroid assignment and centroid updates. Preserve the
existing Lloyd implementation, ten initializations and seed 0. The app now explains
those steps and shows group sizes, mean standardized paths and actual member IDs.
The textbook background is `p4dsf/chapters/27_k_means_clustering/` beside this project.

The new method is **Ward hierarchical clustering**, using SciPy linkage on the
same standardized inputs. Start with individual stocks, merge to minimize the
increase in within-group squared variation, and cut the tree at exactly five
groups using `cut_tree`. It is deterministic for a fixed input order; the seed
is irrelevant to Ward. A standard random-forest classifier requires labels, so
it is not used as an unsupervised grouping substitute.

The app draws the **actual saved Ward linkage tree**, shortened to the final 25
branches. Parenthesized labels count stocks in collapsed branches. Singleton
labels identify within-fit input rows, not ticker symbols. Height measures Ward
merge distance, not return or probability. A dashed line marks the five-group
cut; tied heights are explicitly identified if encountered. The shared
`research/grouping_charts.py::hierarchy_figure` can be called by a notebook/QMD
without source data or fitting. Membership tables show stock IDs separately.

References: [scikit-learn clustering](https://scikit-learn.org/stable/modules/clustering.html),
[SciPy linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html),
[tree rendering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html).

## Same strategy after either grouping

Full-group portfolios compound the same 1/2/3-month earned-return window. Groups
with non-positive momentum get no stock capital. Inside positive groups, only
strictly positive-momentum stocks get weights, proportional to momentum. A group
with P qualifying names out of N names in the book invests P/N of total capital;
the residual stays in zero-return cash. This preserves the original accounting.

Top-ten portfolios first keep up to ten stocks per group ranked by trailing
12-month positive volatility, `sqrt(mean(max(r, 0)**2))`, then recompute group
momentum on that retained universe and apply the same rule. Saved horizons are
one and two months; three-month top ten remains unavailable for both methods.
Equal weight and ungrouped momentum/cash are identical shared controls.

Adding a direction chooses grouping method, classifier, full/top-ten book and
momentum horizon. Both methods reuse the same four existing Basic ML classifiers'
point-in-time forecasts, separately fitted for up and down events. Thus their
full-universe prediction scores are identical; their starting weights and portfolio
results can differ. Downside uses `w*(1-0.5*p_loss)` and leaves the reduction in
cash. Upside boosts by 50% when `p_up >= 0.6`, then rescales to the original stock
budget. Excluded/negative-momentum names stay at zero; no combined overlay exists.

## Saved experiment and publication

- K-means remains `baskets_20261110T021017_e6fa43e2`, unchanged.
- New Ward base: `baskets_20261110T122422_88c2cc4a`.
- New Ward directional replay: `baskets_20261110T122521_a62c9b0c`.
- Forecast sources: downside `20261109T185402_0722a7ec725c`, upside
  `20261109T185621_fe4017faa327`; **zero new directional classifier fits**.
- Ward contains 49 books / 2,940 earned-month rows, 1,988,037 assignment rows
  and 71,448 saved tree merges. Source hashes, grouping settings/code hash and
  directional provenance are recorded in the study metadata.
- Default `published/project`: **169,973,507 bytes**, 458 files. Adds Ward under
  optional schema-1 `tabs.advanced_ml.grouping_methods.ward`; the existing root
  advanced record remains K-means for backward compatibility. All 362 prior
  Parquet files and their checksums are preserved. Details load by
  model/horizon/version. No CSV, catalog or estimator objects are needed.
- Prior default release is preserved locally at
  `published/project_before_grouping_20261110`, excluded by `.gitignore`.

Explicit recomputation, if desired (not an app viewing operation):

```bash
python3 run_basket_momentum.py --grouping ward
python3 run_basket_adjustments.py --basket-study PATH_PRINTED_ABOVE \
  --loss-run 20261109T185402_0722a7ec725c \
  --upside-run 20261109T185621_fe4017faa327
python3 publish_results.py --extend-publication published/project \
  --ward-study ADJUSTED_STUDY_PATH --output published/NEW_RELEASE
```

Exports reject an existing destination. Extension validates training dates,
shared grouping/strategy settings, forecast provenance and exact shared controls,
and verifies every old published frame before copying. Old studies missing the
method field mean K-means. Old hosted releases show an explicit missing Ward state.

Read-only reuse in notebooks/Quarto:

```python
from published_results import PublishedStrategies
from research.grouping_charts import hierarchy_figure
publication = PublishedStrategies('published/project')
ward = publication.baskets().for_method('ward').study()
ward['metrics']                         # saved CAGR, log growth, risk, cash, etc.
hierarchy_figure(ward['hierarchy'], '2009-11-30', n_clusters=5)
```

## Findings, verification and next step

For one-month top ten, K-means CAGR is 24.68%, Ward 21.24%, equal weight 7.62%.
These are gross training-validation results, not a new final selection. Ward is
not automatically better. The app retains both methods and shows a matched
original-portfolio comparison table under Adding a direction.

**210 tests passed** (118.679 s). New tests cover exact tree-cut membership,
future-label/feature-rank invariance, incomplete/constant histories, read-only
fit refusal, saved frames/old-release preservation and both methods' UI. Real-data
verification reconciles all 90 Ward detail partitions, all metrics/monthly/scores/
trees and every prior file checksum. A standalone code-plus-publication copy,
without any CSV or research archive, exercises all six main tabs, both methods,
four classifiers, both starting books and inspection under failing CSV/catalog/
fit/replay guards. Browser checks cover all three subtabs, explanations, the
actual tree and Ward directional choices. See
[verification](verification/grouping_20261110.json) and `grouping_*_20261110.png`.

Feature selection, tuning and final test retain their existing **K-means** sources;
no Ward feature study, parameter search or official-test evaluation was performed.
The formal submission pin, notebook, QMD/HTML and slides are unchanged. Shared
readers/figures above support later integration. Concurrent edits in `context/`
and `research/grouping.py` belong to the other LLM; they were preserved.

Next: review the training comparison in the app, push the code and updated
`published/project` to refresh the hosted site, then decide whether Ward merits a
separate feature-selection/final-evaluation experiment. Do not relabel existing
K-means downstream evidence as hierarchical results.
