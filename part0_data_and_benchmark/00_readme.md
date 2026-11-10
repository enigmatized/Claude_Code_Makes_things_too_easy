# Part 0 — understand the data and earn the benchmark

This is the foundation for the rest of the project. Read the files in order.

| File | Purpose |
| --- | --- |
| [01_dataset.md](01_dataset.md) | What a row means, dates, ranks, labels, and what is unknown. |
| [02_data_exploration.ipynb](02_data_exploration.ipynb) | Executed notebook: audit and training-period charts, without fitting models. |
| [03_equal_weight_benchmark.ipynb](03_equal_weight_benchmark.ipynb) | Executed notebook: worked example, independent calculations, simple/log returns, risk, and benchmark results. |
| [04_pipeline_guide.md](04_pipeline_guide.md) | Reuse the small pipeline when adding strategies or models. |
| [05_strategy_ideas.md](05_strategy_ideas.md) | A short menu of later experiments, linked to course chapters where relevant. |

Browser copies: [data exploration](outputs/02_data_exploration.html) and
[equal-weight benchmark](outputs/03_equal_weight_benchmark.html). Figures are also saved individually
in `outputs/figures/`. The notebooks contain the code that creates these outputs.

`pipeline.py` is the shared calculation code. It has no model or feature selector.
Run its checks from this directory with `python3 -m unittest -v test_pipeline.py`.
Both notebooks work when launched from this folder, the project folder, or the
course folder. They need only the packages already used in this workspace:
NumPy, pandas, matplotlib, Jupyter, and nbconvert for HTML export.

Part 1 and the assignment notebook were left intact. The recorded hashes in
`outputs/preservation_check.json` verify this. No flashcard or presentation work
is included in this foundation.

The benchmark has a fixed definition, so its test-period result can be checked
without choosing a model. Detailed asset exploration uses training data only.
Test returns have already been inspected in earlier work; later strategy results
on that same period should not be described as a newly untouched holdout.

There is no need to infer a hidden trick from the professor's warning. The
benchmark notebook makes the assumptions and common calculation errors visible.
The professor's reference number was not supplied, so our value is a reproducible
answer under the assignment's monthly-rebalancing interpretation, not a claim
that we have independently obtained the grading key.

The executed benchmark gives the following gross, monthly-rebalanced results:

| Earned-return window | CAGR | Annualized volatility | Sharpe, risk-free = 0 | Maximum drawdown |
| --- | --- | --- | --- | --- |
| Training: Jan 2000–Dec 2009 | 14.50% | 19.92% | 0.783 | -49.51% |
| Testing: Jan 2010–Dec 2018 | 13.10% | 15.63% | 0.868 | -20.64% |

The benchmark notebook explains the formulas and provides independent checks;
these figures are not inputs used to force the calculation to a desired answer.
