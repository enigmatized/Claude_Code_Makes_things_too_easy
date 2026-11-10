# Which period is each stage showing?

| Period | Earned returns | Purpose |
| --- | --- | --- |
| Full training dataset | Jan 2000–Dec 2009 | Historical inputs for fitting and research validation. |
| Default training validation | Jan 2005–Dec 2009 | Chronological comparisons after five initial years of history. |
| Test dataset | Jan 2010–Dec 2018 | Later evaluation; this period was inspected in earlier work. |

## Five stages with separate selections

1. **Strategies** shows only original non-ML cases from a training-validation run.
2. **Basic ML** uses that training selection, adds prediction quality and compares
   original/adjusted cases. Its own model filters do not change the source run.
3. **Advanced ML strategy** is an undefined placeholder.
4. **Feature selection** reads saved training-only studies for supported selected
   strategies. Missing experiments and tuning remain clearly pending.
5. **Final test** is separate from training state. Its primary result requires a
   deliberately pinned frozen-model test run that passes the existing submission
   readiness checks. It currently displays a pending status.

Period banners and chart titles use the recorded dates of each run/study. A
training chart does not concatenate test returns and therefore needs no internal
train/test boundary. Future combined charts must label both periods and their
boundary explicitly.

## Validation predicts later observations

The first default decision is Dec 2004 and earns Jan 2005's return. Models use
only outcomes already known at that historical origin, rather than evaluating
predictions on their own fitting rows. Later expanding fits can use earlier
validation outcomes after they become known. Selection/preprocessing stay inside
the corresponding historical training pipeline. The frozen validation option
fits once at the origin instead.

The default non-ML comparison uses the same 2005–2009 months for fairness. A
full-training-period chart would be a distinct labeled comparison with its own
warm-up rules; the UI refactor does not create that experiment.

## Loading and computing

The sidebar lists saved **training** experiments and the shared strategy subset.
**Load training results** changes only training state. Each training stage has
an explicit compute expander; editing settings alone does not replace results.
Computation is restricted to training validation in those controls and exact
saved matches are reused first.

A saved computation can contain both original and adjusted cases. Strategies
filters that recorded experiment to originals without rewriting its config,
manifest or scientific provenance. Basic ML reads model cases from the same
source. Contextual run reports export the complete source rather than pretending
that a display filter was a new computation.

The two momentum-study views have independent widget state. Strategies shows
only the four original windows; Basic ML can additionally show model versions.
Their allocation loader updates training selection only. Feature selection's
saved study has its own clearly labeled validation period.

## Final and historical test evidence

**Final test** does not select the newest test run automatically. A frozen model
alone does not establish final readiness: feature limits, tuning evidence,
benchmarks, risk/reward measures and a justified pinned final choice must also
be recorded. Official test models fit before evaluation; lagged non-ML signals
can update as returns become known.

Enable **Show historical / exploratory test evidence** to inspect thirteen
previously saved non-ML 2010–2018 portfolios. Their CSV histories are read directly,
with overlapping strategies reconciled and training rows excluded. These archives
exclude the later annual-consistency cash strategy and its lookback variants.
They are not new current-code experiments or the final chosen model's test result.

Registered saved tests, when available, can be loaded as exploratory evidence in
this last tab. That callback updates `test_result`, never `training_result`.
Expanding-test refits are explicitly exploratory; earlier test labels may enter
later fits. Frozen fits remain identified by their own recorded protocol.

No new research test experiment was performed for this refactor. The earlier
inspection of test data remains disclosed; do not describe it as untouched.

## Implementation

`result_views.py` supplies immutable case-filtered views and the pinned-final
readiness gate. `research/evaluation.py` supplies date/protocol descriptions and
historical archive reads. `evaluation_ui.py` renders those existing records.
Scientific computation, saved artifacts and the publication release are unchanged.
