# Momentum acceleration experiment — 2026-11-10

Open **Grouping With ML → Momentum acceleration** for training results, or
**Final test → Momentum acceleration** for the separately labeled 2010–2018
check. Both research and deployed views read the same completed experiment.
The original K-means and Ward strategies and directional models are preserved.

## What the original basket actually does

At each month-end, K-means or Ward groups stocks using twelve already-earned
monthly returns, standardized within each stock. Five groups are requested.
Within each group, retain up to ten stocks ranked by twelve-month positive-side
volatility, `sqrt(mean(max(return, 0)**2))`. This is an upside semideviation, not
the standard deviation conditional on a positive month.

Recompute each selected group's one-month momentum as the **equal-weight mean**
of its members' latest earned returns. The group must have positive momentum;
each held member must also have positive individual momentum. If N names passed
the screen across all groups and P qualifying names belong to a positive group,
that group's invested fraction is P/N. Divide that fraction among its qualifying
names **in proportion to their positive momentum**. Remaining capital is cash.

For example, four screened names, with +10%, +30% in a positive group and two
negative names in another group, produce weights 12.5%, 37.5%, 0%, 0%, and
50% cash. Group return signals are equally weighted; actual capital is not.
This task does not replace the original with an equally allocated basket strategy.

## Declared additional rule

Acceleration is the discrete change in monthly simple return:

`stock_acceleration(t) = earned_return(t) - earned_return(t-1)`.

A stock returning +10% then +4% has acceleration −6 percentage points, even
though its momentum remains positive. Basket acceleration uses the same formula
on equal-weight mean returns of the **current selected members in both months**.
Never subtract aggregates for different memberships or assume group 0 this month
has the same identity as group 0 last month.

Four variants use identical groups, screening and original allocations:

| Variant | Additional multiplier on each original stock weight |
|---|---|
| Original | 1 |
| Stock deceleration | 0.5 when stock acceleration < 0; otherwise 1 |
| Basket deceleration | 0.5 when its basket acceleration < 0; otherwise 1 |
| Stock + basket deceleration | Product of those two multipliers |

Both negative signals retain 25% of the original position. Reductions go to
zero-return cash; never renormalize or introduce negative positions. A zero
original weight stays zero. Missing comparisons are retained as missing signals
and receive a neutral multiplier of 1, rather than inventing observations.

The 50% haircut, one-month momentum and two-month comparison were recorded in
`study.json` before computation. No parameter search or classifier fitting is
involved. This noisy signal can duplicate momentum or discard useful winners;
it is an experiment, not a guaranteed improvement. This is not yet an ML input
recipe: adding acceleration to classifiers requires separate training/CV and
comparison with the same frozen rules.

## Evidence and timing

Study: `acceleration_20261110T204704_5c0c838f`.
Training earns January 2005–December 2009 (60 months), and its stage is computed
before opening the test CSV. The separate final stage earns January 2010–December
2018 (108 months). This test period was already viewed elsewhere in the project;
do not describe it as untouched or as independent confirmation of a new choice.
Groups and signals can update with already-earned returns. Stored snapshot-t
labels earn in t+1, so neither signal uses the label on its decision row.

Each stage saves eight metric rows (two grouping methods × four rules), plus
480/864 monthly rows and 6,000/10,800 selected-stock audit rows. The original
training/test monthly books match all existing K-means and Ward reference books
exactly, including returns, wealth, drawdown, cash and turnover.

Illustrative K-means results; gross of costs, cash return 0%:

| Window / variant | CAGR | Sharpe | Maximum drawdown | Mean cash |
|---|---:|---:|---:|---:|
| Training / Original | 24.68% | 0.833 | −20.55% | 50.43% |
| Training / Basket deceleration | 23.58% | 0.826 | −18.88% | 58.50% |
| Previously viewed test / Original | 19.61% | 0.591 | −21.33% | 51.56% |
| Previously viewed test / Basket deceleration | 14.36% | 0.669 | −14.65% | 58.88% |

The smaller drawdown comes with lower CAGR and more cash. Do not call it a clear
improvement on every objective or select a preferred rule using these test rows.
All four variants, both methods, CAGR/log returns and inspection audits appear
in the app. Basket return bars compare the same current members in both months.

## Code, saved data and publication

- `research/momentum_acceleration.py`: causal signals and fixed multipliers;
  reuses grouping, stock screening, momentum allocation and cash accounting.
- `run_momentum_acceleration.py`: explicit offline computation only. Writes an
  immutable study plan and checksummed training/test frames under ignored
  `artifacts/acceleration_studies/`; blocked by `MLFACTOR_READ_ONLY=1`.
- `acceleration_publication.py`: explicit immutable export and a manifest-backed
  reader. Adds optional `tabs.acceleration`, preserving all prior frame hashes.
- `acceleration_ui.py`: shared local/published saved-results renderer. Training
  and final readers validate declared dates and one-month earning alignment.

To recompute deliberately in research mode:

```bash
python3 run_momentum_acceleration.py
```

To add a completed study to a new publication directory:

```bash
python3 publish_results.py --extend-publication published/project \
  --acceleration-study artifacts/acceleration_studies/acceleration_20261110T204704_5c0c838f \
  --output published/my_new_acceleration_release
```

Current `published/project` adds only six display Parquet files, approximately
0.76 MB. It has 476 files / 173,332,889 bytes, preserving 469 previous frame
hashes. Its prior default remains in ignored
`published/project_before_acceleration_20261110`. No new dependencies or hosting
settings. Existing `.gitignore` includes the current release while excluding
source CSVs, research artifacts and publication backups.

For a read-only notebook/QMD cell, use the same reader, without training:

```python
from published_results import PublishedStrategies
reader = PublishedStrategies(PROJECT_ROOT / 'published/project').analysis_section('acceleration')
training = reader.study('training')
training['metrics']  # same metrics and monthly/signal tables used by the app
```

The formal notebook/report/slides/release pin are not rewritten by this app
experiment. Another LLM is editing the submission notebook independently;
preserve that work. Verification lives in
`docs/verification/momentum_acceleration_20261110.json` and the companion PNGs.
Next possible experiment: a separately declared ML input recipe with stock/basket
acceleration, chronological fitting and fixed original comparison. No such
classifier result is claimed here.
