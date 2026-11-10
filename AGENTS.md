# Project handoff and maintenance

All six tabs now support the
self-contained published-results mode. See `docs/DEPLOYMENT.md` before extending
another tab. Keep local research available, deployment strictly read-only, and
first five tabs strictly training-only. Final test has separately labeled frozen
test/training/entire-path windows. Advanced ML has two subtabs: build the grouped
strategy, then add directional predictions. Deployment must never open research
archives or source/result CSVs; extend the explicit manifest exports instead.
Feature selection, Tuned Model and Final test default to four portfolio families;
turn their focus filter off to review additional saved comparisons. Classifier
algorithms and portfolio families are separate concepts. Advanced tuning has
not been run; the focused tuning view honestly has equal-weight evidence only.

Current advanced-stage authority: the user subsequently defined correlation
groups, top-positive-volatility screening and momentum, then requested separate
downside-to-cash and upside-weight overlays. These are implemented; preserve them.
See `docs/BASKET_MOMENTUM.md`. Older placeholder instructions below record the
earlier scope and do not require removing these separately authorized additions.

Read these before changing this project:

1. `docs/PROJECT_STATE.md` — current implementation, evidence, unfinished work.
2. `docs/ARCHITECTURE.md` — computation, saved results, UI and submission boundaries.
3. `docs/SUBMISSION.md` — requirements verified against the assignment notebook.
4. `docs/DECISIONS.md` and `CHANGELOG.md` — accepted decisions and recent changes.

For app navigation work, also read `docs/TAB_REFACTOR_HANDOFF.md`. It records the
latest requested five-tab structure and supersedes earlier navigation proposals.
Its advanced-ML tab is intentionally undefined. The subsequent user request
"Implement please" authorized the five-stage refactor, which is now implemented.
Keep that advanced slot a placeholder until the user separately defines it.
The separately authorized 2026-11-09 upside target/per-stock comparison belongs
within Basic ML; see `0_6_Basic_ML_Directions/README.md`. It does not define the
advanced strategy. Preserve the loss/pooled default's numerical parity.

After every substantive code, data, configuration, experiment or report update:

- Update PROJECT_STATE with what works, what remains, and verification evidence.
- Append a dated CHANGELOG entry; update DECISIONS when a design choice changes.
- Update the relevant usage/architecture/submission documentation and links.
- Record actual experiment provenance and distinguish computation from cached reads.
- Leave a short next-step handoff. Never claim a pending experiment was completed.

Keep source CSVs, the assignment, original Part 1, and earlier experiment outputs.
Reuse `research/` from notebooks, the app and Quarto; do not copy accounting or ML
training into presentation code. Rendering reports or browsing results must not
fit models. Prefer existing saved results; recomputation is an explicit action.

All stock-month joins use date and stock_id. The label on snapshot t is earned
in t+1. Never use it as a same-row predictor. Calendar gaps break return-history
eligibility. The test period was already viewed; do not call it untouched.

The final assignment requires <=10 features, four model types, training-only
GridSearchCV tuning where applicable, two non-ML comparisons including equal
weight, and risk/reward analysis. Submission models must be fitted using training
data; keep exploratory expanding test refits distinct from frozen submission fits.

Follow the existing tests relevant to a change. Preserve exact default logistic
behavior and original portfolio results unless an intentional change is clearly
documented. Add meaningful checks for timing, cache identity, and report provenance.
