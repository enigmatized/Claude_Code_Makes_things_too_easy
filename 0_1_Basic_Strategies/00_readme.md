# Basic non-ML strategies

Read [01_strategy_rules.md](01_strategy_rules.md), then run
[02_basic_strategy_comparison.ipynb](02_basic_strategy_comparison.ipynb).
The executed browser copy is [outputs/02_basic_strategy_comparison.html](outputs/02_basic_strategy_comparison.html).
[03_findings.md](03_findings.md) is generated from that run.

For the follow-up combining a **frozen training high-volatility list** with
monthly momentum allocation, read [04_hybrid_rules.md](04_hybrid_rules.md) and
[05_high_volatility_momentum.ipynb](05_high_volatility_momentum.ipynb).
The [hybrid browser copy](outputs/hybrid/05_high_volatility_momentum.html) and
[06_hybrid_findings.md](06_hybrid_findings.md) contain its executed comparison.
Run its additional checks with `python3 -m unittest -v test_hybrid.py`.

For **three-month risk with an extra penalty for negative returns**, read
[07_downside_rules.md](07_downside_rules.md) and
[08_three_month_downside.ipynb](08_three_month_downside.ipynb). Its
[browser copy](outputs/downside/08_three_month_downside.html) and
[09_downside_findings.md](09_downside_findings.md) compare the new hybrid sizing
rule with ordinary three-month volatility and explain Sharpe versus Sortino.

This step follows Part 0. It implements the high-volatility, low-volatility, and
positive-momentum portfolios, plus controls that make their differences easier
to explain. No machine learning, original-factor selection, or parameter search
is performed. Part 0 and Part 1 are preserved.

| Portfolio | Question |
| --- | --- |
| Equal-weight universe | What does the whole available market do? |
| Equal-weight eligible stocks | What changes just from requiring historical data? |
| High-volatility basket | What happens when we concentrate in volatile stocks? |
| Low-volatility basket | What happens when we concentrate in quiet stocks? |
| Positive-momentum basket, equal weights | What happens when we select recent winners? |
| Same momentum basket, momentum weights | What changes when stronger momentum gets more capital? |

Signals use twelve consecutive realized monthly returns. Baskets target 20% of
eligible stocks and rebalance monthly. These are fixed introductory choices,
not optimized values. The two momentum books hold identical names unless both
fall back; their difference is capital allocation.

The first twelve training return months supply history. Consequently the fair
training comparison earns Jan 2001–Dec 2009, while the test comparison retains
Jan 2010–Dec 2018 using earlier history. An additional CSV retains the original
full-window equal-weight benchmark for reference. It is not mixed into the
matched-window training table.

## Files and rerunning

`strategies.py` builds signals and weights and reuses Part 0's `pipeline.py` for
returns, risk, and drift-aware turnover. The notebook explains the experiment,
writes outputs, and creates figures. Start it from this folder, the project
folder, or the course folder. Run checks from this folder:

```bash
python3 -m unittest -v test_strategies.py
```

`outputs/` contains monthly returns, comparison metrics, yearly results,
allocation diagnostics, source hashes, and PNG/SVG figures. Compressed signals
and keyed weights allow a later notebook to inspect a particular rebalance
without reconstructing the experiment. They are not selected original factors.

Gross results retain supplied extreme returns and use risk-free return zero.
Momentum score weights are deliberately uncapped; concentration is reported.
There is no assumption that high individual-stock volatility guarantees a
higher-volatility portfolio. Test returns were inspected in earlier project
work. These are historical rule comparisons, not a new untouched validation set.
