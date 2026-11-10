# Part 1 plan: choose at most 10 features and test the long-only book

Status: method locked on 2026-10-24, before `02_part1_feature_selection.ipynb` exists.

This file is the specification for assignment item 1 in `03_mlfactor_long_only_project.ipynb`:

> Perform feature subset selection where you choose 10 or less features. Your feature selection process should involve data analysis and/or machine learning techniques. Conduct some analysis as to whether restricting to your feature subset seems to be improving strategy performance.

The notebook that follows this plan has to teach the steps, not only print a feature list. A later reader should be able to say what each table means, why the subset is that size, and whether the long-only book got better.

Part 1 stops at that question. These assignment items stay out of this notebook:

- Item 2, four model families.
- Item 3, `GridSearchCV` and the study of machine-learning metrics versus strategy performance.
- Item 4, the second non-ML strategy. The equal-weight universe is computed here as a reference line so the plots have a baseline.
- Item 6, the final model choice across every experiment.

---

## 1. The decision this part has to earn

There are 20 factor columns. The delivered object is a frozen list, `subset_final`, of those original columns, length 3 to 10.

The list is then judged by a pre-committed trading rule: each month, hold an equal-weight book of the stocks in the top quintile of predicted next-month return. The same rule, the same model, and the same hyperparameters are run twice, once on all 20 columns and once on `subset_final`. The only thing that changes is the column set.

"Improving strategy performance" means that comparison. A higher in-sample R² is not the answer. A prettier feature-importance bar chart is not the answer.

The decision rule, applied after the test backtest exists and not before:

- Call the subset an improvement when, on the test window, the Ridge subset book has a higher textbook Sharpe than the Ridge all-20 book, the paired monthly return difference is positive, and the same direction shows up in the yearly walk-forward inside the training years.
- Call it a simplification with similar performance when absolute Sharpe difference is under 0.05 and absolute annualized-return difference is under 1 percentage point. Similar performance with fewer redundant inputs is a useful result. The assignment asks whether restricting helps, and "it holds up while dropping near-duplicate factors" is a real answer.
- Call it no clear improvement when the test comparison and the walk-forward disagree, when the ranking flips after the outlier cap in section 8, or when the Sharpe gap is positive and the walk-forward gap is not.
- The feature list is not edited after any test-window number is computed.

---

## 2. What is in the files

Both CSVs have the same columns.

Identifiers and target:

- `date` — month-end snapshot. The factors are known on this date.
- `stock_id` — security id.
- `R1M_Usd` — the stock's return over the month after `date`. This is the label. It is also the return the backtest is allowed to earn after the portfolio is chosen.

The other 20 columns are the features.

Measured while writing this plan. The notebook recomputes all of these and treats a mismatch as something to explain in `03_learnings_from_implementation.md`.

| | Train | Test |
| --- | --- | --- |
| File | `mlfactor_long_only_train.csv` | `mlfactor_long_only_test.csv` |
| Rows | 140,739 | 127,740 |
| Snapshots | 120 | 108 |
| First snapshot | 1999-12-31 | 2009-12-31 |
| Last snapshot | 2009-11-30 | 2018-11-30 |
| Distinct stocks | 1,204 | 1,195 |
| Stocks in both | 1,193 |  |
| Duplicate `(date, stock_id)` | 0 | 0 |
| Missing values | 0 | 0 |
| Names per month | about 1,050 to 1,200 | about 1,110 to 1,190 |

Calendar, because the graded equal-weight return is easy to shift by a month:

- The training snapshots run December 1999 through November 2009. Their labels are the returns earned January 2000 through December 2009. That is the prompt's "2000-2009" training period.
- The test snapshots run December 2009 through November 2018. Their labels are the returns earned January 2010 through December 2018. That is the prompt's "2010-2018" test period.
- The last training label is December 2009. The first test label is January 2010. The two files do not share a return month.
- The December 2009 snapshot stays in the test file. Dropping it would delete the January 2010 return.

Cross-section size changes through time. Each month is scored on the stocks present that month. There is no requirement that a test stock appeared in the training file.

### The features are already percentile ranks

Every feature lies on `[0.01, 1.00]`, with mean about 0.50, median 0.50, standard deviation about 0.289, and about 100 distinct values. The standard deviation of a uniform `[0, 1]` random variable is `1/sqrt(12) ≈ 0.289`. These columns behave like cross-sectional percentile bins.

Chapter 40 (`p4dsf/chapters/40_numerai`) describes the same shape and treats it as data that is already normalized. Two consequences:

- A high value means "high raw characteristic relative to the other stocks that month," as the vendor ranked it. It does not mean the characteristic is attractive. Dividend yield and EV/EBITDA can both be stored as high-is-large. The model is allowed to learn a negative sign. Nothing in the pipeline flips a column by hand.
- KNN-style scaling is not the problem it was in the k-nearest-neighbors project. Linear shrinkage still gets a `StandardScaler` fit on the training rows of that fit, because chapter 23 scales features before ridge and lasso so the penalty treats them in comparable units. On these columns the scaler barely moves anything. Trees are fit on the raw percentiles.

Working names, used for interpretation after the list is chosen. The selection math does not depend on these names being right.

| Column | Working meaning | Name confidence |
| --- | --- | --- |
| `Ocf_Toa`, `Ocf_Ta` | Operating cash flow over assets | Medium. The two names may be duplicates or near-duplicates. |
| `Ocf_Ce` | Operating cash flow over common equity, or cash earnings | Low. |
| `Ocf_Tbv`, `Fcf_Tbv` | Operating or free cash flow over tangible book | Medium. |
| `Fcf_Yld` | Free-cash-flow yield | High. |
| `Div_Yld` | Dividend yield | High. |
| `Bb_Yld` | Buyback yield | Medium. |
| `Netdebtyield` | A net-debt yield or related valuation | Low. |
| `Ev_Ebitda` | Enterprise value / EBITDA | High. |
| `Tot_Debt_Rev` | Total debt / revenue | High. |
| `Net_Debt` | Net debt, ranked across stocks | Medium. |
| `Debtequity` | Debt / equity | High. |
| `Total_Debt_Capital` | Debt / total capital | High. |
| `Total_Liabilities_Total_Assets` | Liabilities / assets | High. |
| `Tev_Less_Mktcap` | Enterprise value minus market cap, ranked | Medium. Closely related to net debt. |
| `Share_Turn_3M`, `Share_Turn_6M`, `Share_Turn_12M` | Share turnover over 3, 6, and 12 months | High. |
| `Capex_Sales` | Capex / sales | High. |

### The label has a thin signal and a fat tail

Train `R1M_Usd` has mean about 0.013, median 0.011, and a maximum of 27.2. The test maximum is about 30.2. On the training file, 146 rows with absolute return above 1 account for about half of the sum of squared returns. A squared-error fit on raw returns spends its energy on those rows.

That is why section 6 winsorizes the fit target and why the association measure is a within-month Spearman correlation. Spearman uses ranks, so one 27x return cannot decide which feature looks predictive.

Scouting monthly Spearman information coefficients on the training months (mean across 120 months, t-stat = mean / (std / sqrt(120))). The notebook recomputes this table. It is train-only.

| Feature | Mean IC | t-stat |
| --- | --- | --- |
| `Ocf_Ce` | 0.011 | 1.78 |
| `Fcf_Yld` | 0.010 | 1.37 |
| `Share_Turn_6M` | −0.009 | −0.60 |
| `Share_Turn_12M` | −0.009 | −0.57 |
| `Ocf_Ta` | 0.008 | 1.02 |
| `Ocf_Toa` | 0.008 | 1.17 |
| `Share_Turn_3M` | −0.008 | −0.52 |
| `Ocf_Tbv` | 0.007 | 1.55 |
| `Netdebtyield` | −0.005 | −1.16 |
| `Tot_Debt_Rev` | −0.005 | −0.50 |
| `Capex_Sales` | −0.005 | −0.70 |
| `Ev_Ebitda` | −0.004 | −0.46 |
| `Tev_Less_Mktcap` | 0.003 | 0.28 |
| `Bb_Yld` | −0.003 | −0.54 |
| `Total_Debt_Capital` | 0.003 | 0.27 |
| `Fcf_Tbv` | 0.003 | 0.63 |
| `Debtequity` | 0.002 | 0.20 |
| `Net_Debt` | −0.002 | −0.14 |
| `Total_Liabilities_Total_Assets` | −0.001 | −0.10 |
| `Div_Yld` | 0.000 | −0.01 |

No training IC clears a conventional t-stat of 2. An information coefficient near 0.01 is ordinary for a monthly equity factor. The subset is a way to stop a 20-column model from fitting noise and from double-counting one economic idea. It is not a claim that a strong factor is hiding in the list.

A test-set information-coefficient table was computed while scouting the files. Those numbers are omitted on purpose. They play no role in the rule below.

### Several columns are the same idea measured more than once

Pearson correlations on the training rows:

- `Share_Turn_3M`, `Share_Turn_6M`, `Share_Turn_12M`: 0.95 to 0.98.
- `Debtequity` and `Total_Debt_Capital`: 0.97. `Tot_Debt_Rev` correlates about 0.84 with both.
- `Fcf_Tbv` and `Ocf_Tbv`: 0.86.
- `Net_Debt` and `Tev_Less_Mktcap`: 0.74.
- `Ocf_Toa` and `Ocf_Ta`: 0.65. Related, and still distinct at a 0.80 cutoff.

Average-linkage clustering on distance `1 - |correlation|`, cut at distance 0.20 (that is, `|correlation| = 0.80`), produced these training clusters:

- (`Tot_Debt_Rev`, `Debtequity`, `Total_Debt_Capital`)
- (`Share_Turn_3M`, `Share_Turn_6M`, `Share_Turn_12M`)
- (`Fcf_Tbv`, `Ocf_Tbv`)
- every remaining column in its own cluster

Cutting at distance 0.30 (`|correlation| = 0.70`) also merges `Total_Liabilities_Total_Assets` into the leverage cluster and merges `Net_Debt` with `Tev_Less_Mktcap`.

There are 15 clusters at the 0.80 cut and only 10 seats. Correlation screening is required, and it is not enough. The score in section 6 has to leave some weakly related columns out.

Chapter 26a makes the same point with principal components: when a block of columns moves together, the first component soaks up the shared variation, and treating the columns as independent overstates how much information you have. PCA is used in the notebook as a picture of that fact. The submitted list stays in the original columns, because the assignment asks for a feature subset of the factor measurements.

---

## 3. Why a smaller list can help this specific book

Three mechanisms, each tied to something already measured:

1. **Double counting.** A linear model that is handed three turnover columns with correlation 0.97 has to split one piece of information across three noisy coefficients. Lasso, which chapter 23 and chapter 34 use for this reason, drives some of those coefficients to exactly zero. Ridge shrinks them and keeps all three. The all-20 strategy in this part uses ridge, so it is the version that has to live with the duplicates. The subset is the version that keeps one representative.

2. **Squared error chases the tail.** Half of the training sum of squares sits in 146 extreme returns. Extra columns give that tail more ways to leak into the fit. Winsorizing the fit target reduces the damage. Using fewer columns reduces it again. Chapter 29 is the course's overfitting chapter: a flexible fit looks better on the sample it was built on and worse on later data. Here the "later data" is the next month's cross-section.

3. **The trade is a rank, and the rank is weak.** The book only needs an ordering inside the month. A column that is redundant, or whose rank correlation flips sign across eras, adds variance to that ordering. Keeping the columns that survive several training eras is the stability check.

A smaller list can also throw away a real, small, nearly orthogonal signal. `Capex_Sales`, `Div_Yld`, `Bb_Yld`, and `Ev_Ebitda` are not in the tight clusters. They get a chance on their own scores. The cap of 10 is large enough to keep several of them if the evidence says so. The rule refuses to pad the list up to 10 when the evidence is empty.

---

## 4. What comes from the textbook, and what is added

Used as the course teaches them:

| Idea | Where |
| --- | --- |
| Lasso sets coefficients to zero; ridge shrinks them and leaves them nonzero | Ch. 23 |
| Scale features before a shrinkage penalty | Ch. 23 |
| Read a lasso path, then drop a column whose coefficient is zero | Ch. 34 |
| A backtest of the fitted model, compared with a simpler benchmark | Ch. 33–34, and the train/test split in ch. 25 |
| `feature_importances_` as a nonlinear second opinion | Ch. 29 |
| PCA as a description of collinear blocks | Ch. 26a, 26b |
| Features already on `[0, 1]` with mean 0.5 and std about 0.28 | Ch. 40 |
| Annualized compound return `((1+r).prod())**(k/len(r)) - 1`, volatility `r.std()*sqrt(k)`, Sharpe `(r.mean()/r.std())*sqrt(k)` | Ch. 15b, with `k = 252` for daily data |
| Equity curve and max drawdown `(equity / running_peak) - 1` | Ch. 16 |
| Correlation heatmaps | Ch. 12 |

Added because this file is a stock-month panel and the textbook examples are mostly a single table of independent rows:

1. **Splits are by month, in time order.** Chapter 31 and chapter 34 use `KFold` and a random-looking cross-section. Rows in this file share a month, and months share regimes. A random fold puts 2008 into both sides of a fit. Every split in this part is a cut in the list of snapshot dates.

2. **The fit target is a within-month winsorized, demeaned return.** Chapter 25 predicts the next raw return of one stock, which is the right target for that chapter. Here the book is long-only inside a cross-section, so the model should learn relative performance. Section 6 writes the exact transform.

3. **The lasso's alpha is chosen by within-month Spearman correlation on a later block of months.** Chapter 34 looks at R² and finds alpha hardly matters. Spearman correlation is the quantity the trading rule uses. An MSE-chosen alpha is computed and stored on the scorecard so part 3 has something to compare. It does not enter the score.

4. **Selection frequency across expanding eras.** One lasso fit on all ten training years hides instability. A column counts only to the extent it is nonzero in later eras as well. This is the practical form of stability selection. The implementation is a loop, not a new estimator class.

5. **Permutation importance measured as the drop in monthly Spearman correlation, shuffling inside the month.** Chapter 29 reads impurity importance. Impurity importance favors columns the tree happened to split on. Permuting one column inside each validation month, then watching the rank correlation fall, asks the question the strategy asks: did this column's cross-sectional ordering matter?

6. **A fixed shallow random forest is the second strategy model on the test window.** It is there so a subset that helps only because ridge hates collinearity can be seen as such. Its hyperparameters are written in this file and are not searched. Searching them is item 3.

7. **Yearly rebuilding of the subset on past months only.** That produces a 60-month horse race inside 2005–2009 in which the feature list never sees the months it then trades. The official list for the test decade is still fit once, on the full training file, which is what the prompt describes.

8. **A 10 bp turnover haircut and a capped-return view of the same backtest.** The data have no spread or impact fields. These two views show whether one comparison is an artifact of a 27x print or of a book that replaces every name every month.

XGBoost is installed (3.4.1) and chapter 32 uses it. In this part its gain importance is a diagnostic column on the scorecard. It does not enter the score and it is not one of the four required model families. Those belong to a later part. Putting it in the score would add a third knob after the design was supposed to be frozen.

---

## 5. Rules the notebook follows

- Sort by `date`, then `stock_id`, at load.
- `date` and `stock_id` are never features.
- `select_features` receives a frame of training snapshots and returns a list. It reads no test row.
- `subset_final = select_features(all 120 training months)` is written to `subset_final.csv` before the test file is used in any model, metric, or plot.
- The test backtest is one pass. Plots and the outlier cap are views of that pass.
- `random_state=0` everywhere a model accepts it.
- Library stack already in this environment: pandas 2.2, numpy 1.26, scikit-learn 1.4, scipy 1.13, matplotlib 3.8, seaborn 0.13, xgboost 3.4, statsmodels 0.14. The Newey-West number in section 8 is computed directly, so the notebook does not depend on a particular statsmodels API.

---

## 6. The selection procedure

`select_features(df)` uses only the rows in `df`.

### 6.1 Fit target

For each month separately:

1. Clip `R1M_Usd` to that month's 1st and 99th percentiles. With about 1,170 names, each tail is about a dozen stocks. This is the step that stops a single 27x return from owning the squared error.
2. Subtract that month's median of the clipped returns.

Call the result `y_fit`. The clip thresholds and the median are computed on the rows inside the fit being trained. A validation month is transformed with its own clip and its own median when it is itself a fitting sample. When a fitted model is scored, the score uses raw `R1M_Usd`, not `y_fit`.

Predictions are used only to rank inside the month, so the unit of `y_fit` does not change who gets held.

### 6.2 Month folds

Let the unique snapshots in `df` be sorted, `n` of them.

```text
min_train = 48
val_size = 12
start = 48
while start + 12 <= n:
    train_months = snapshots[0:start]
    val_months   = snapshots[start:start+12]
    keep this fold
    start += 12
```

On all 120 training months this yields six folds. The first fit uses the first 48 snapshots (through November 2003). Validation blocks then walk forward 12 months at a time through November 2009. Later folds include the financial-crisis years, so a column that only worked in the early sample loses frequency.

On a shorter frame the same rule yields fewer folds. At the first walk-forward origin (60 months) there is one fold. That is a weaker frequency estimate, and the learnings file says so when it reports that origin.

### 6.3 Lasso frequency

For each fold:

1. Hold out the last 12 months of that fold's training months as the inner block. With a 48-month outer train, the inner fit has 36 months and the inner block has 12. Both stay inside the outer train. The fold's validation months are untouched.
2. Alpha grid: `np.logspace(-5, -2, 16)`.
3. For each alpha, fit `StandardScaler` on the inner-fit rows only. Fit `Lasso(alpha=alpha, max_iter=20000)` of the scaled features on `y_fit`.
4. Predict the inner block. For each inner month, compute Spearman correlation between the prediction and raw `R1M_Usd`. The alpha's score is the mean of those month correlations.
5. Eligible alphas are those within 0.002 of the best score. Among them, take the largest alpha. A flat path then prefers the sparser model. Chapter 34 found a flat path on its own data; the tie-break is here so a flat path has a defined answer.
6. Refit that alpha on every outer-training month of the fold, with a scaler fit on those rows only.
7. A column is selected on the fold when `abs(coefficient) > 1e-8`.

`lasso_frequency` is the fraction of folds that select the column. Also store the mean coefficient across folds, counting a missed fold as zero, and store the alpha chosen on each fold.

Side column, stored and unused by the score: repeat step 3–5 with inner score equal to negative mean squared error on `y_fit`, and record that alpha's selected set as `lasso_mse_frequency`. Part 3 is where machine-learning loss versus rank correlation gets a real experiment. Part 1 only shows whether the two alphas would have pointed at the same columns.

### 6.4 Permutation drop

Same folds. Model, fixed:

```text
RandomForestRegressor(
    n_estimators=200,
    max_depth=3,
    min_samples_leaf=400,
    max_features="sqrt",
    random_state=0,
    n_jobs=-1,
)
```

Depth 3 and a leaf of at least 400 rows are there because the rank correlation is near 0.01. A deep forest would fit month-level noise and the importance numbers would describe that noise. These values are not tuned.

Fit on the fold's outer-training rows. Features are the raw percentiles. Target is `y_fit`.

Baseline: mean, over the fold's validation months, of Spearman(prediction, raw `R1M_Usd`).

For each feature, five times:

- Copy the validation features.
- Inside each validation month, shuffle that one column across stocks.
- Predict and recompute the mean monthly Spearman correlation.
- Drop = baseline − shuffled score.

`perm_drop` is the average of those drops across the five shuffles and across folds. A positive drop means the column's within-month ordering was helping the rank correlation.

### 6.5 Diagnostic columns that do not vote

On the same folds, fit:

```text
XGBRegressor(
    n_estimators=100,
    max_depth=2,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=0,
    n_jobs=-1,
    tree_method="hist",
)
```

Store the average `feature_importances_` as `xgb_gain`. Depth 2 and a small learning rate keep this in the same spirit as the shallow forest. It is a column on the scorecard so a reviewer can see disagreements. It does not enter the score.

Also store, from the whole frame passed to `select_features`:

- `mean_ic`, `std_ic`, `t_ic`, `hit_rate` of each feature against raw `R1M_Usd`, month by month.
- `stability = 2 * abs(hit_rate - 0.5)`. This is 0 when the sign is a coin flip and 1 when every month has the same sign.
- `ic_stat = abs(mean_ic) * stability`.

### 6.6 Score

Min-max scale `perm_drop` across the 20 features, and min-max scale `ic_stat` across the 20 features. If the max equals the min, the scaled value is 0.

```text
score = 0.40 * lasso_frequency
      + 0.40 * perm_drop_scaled
      + 0.20 * ic_stat_scaled
```

The weights are fixed here. Lasso frequency and the permutation drop carry the multivariate evidence. The univariate information coefficient is down-weighted because every one of those t-stats is weak, and because a raw `|mean IC|` would promote a column whose sign flips every year. `stability` is what punishes the flip.

### 6.7 From scores to a list of at most 10

A feature is eligible when at least one of these is true:

- `lasso_frequency >= 0.5`
- `perm_drop > 0`
- `abs(t_ic) >= 1`

Sort eligible features by `score` descending, then by column name ascending so ties do not depend on row order.

Greedy pass, using the Pearson correlation matrix of the frame:

- Take the next feature when its absolute correlation with every feature already taken is strictly below 0.80.
- Stop at 10 names, or when the eligible list ends.

The 0.80 gate is the operational version of the cluster cut in section 2. It keeps one turnover column, one of `{Debtequity, Total_Debt_Capital, Tot_Debt_Rev}`, and one of `{Fcf_Tbv, Ocf_Tbv}`, and it leaves room for the columns that are actually distinct.

If fewer than 3 features are eligible, append the next-highest scores from the full list, still respecting the correlation gate and the cap, until the list has 3 names or no name remains. Set `weak_signal_fallback = True`. A list shorter than 3 makes the strategy comparison a single-factor story that this data's t-stats do not support. A list of 3 to 10 is allowed, including a list that stops at 4 because only 4 were eligible. The assignment says 10 or fewer. Filling empty seats with columns that failed every check would invent a subset.

Write the full scorecard, eligibility flags, and the chosen rows.

### 6.8 Two lists, two jobs

**Process check, inside the training decade.** Origins at snapshot index 60, 72, 84, 96, and 108 (0-based on the sorted training dates):

| Origin index | Last snapshot the selector may see | Months then traded |
| --- | --- | --- |
| 60 | 2004-11-30 | 2004-12 through 2005-11 |
| 72 | 2005-11-30 | 2005-12 through 2006-11 |
| 84 | 2006-11-30 | 2006-12 through 2007-11 |
| 96 | 2007-11-30 | 2007-12 through 2008-11 |
| 108 | 2008-11-30 | 2008-12 through 2009-11 |

At each origin, `subset_t = select_features(snapshots strictly before the origin)`. That subset trades the next 12 snapshots. This path is 60 months long. It is the evidence that the act of restricting the column set helps, because each list was chosen before the year it trades.

**Deployed list.** `subset_final = select_features(all 120 training snapshots)`. This is the list the test backtest and the later parts of the project use. It is allowed to differ from the yearly lists. The learnings file reports the overlap. Large disagreement means the identity of the subset is unstable, which the decision rule treats as "no clear improvement" even if one test number looks good.

---

## 7. The trading rule that both column sets share

The rule is fixed so the horse race is about columns.

**Model for the walk-forward and for the primary test comparison:**

```text
Pipeline([
    ("scale", StandardScaler()),
    ("model", Ridge(alpha=1.0)),
])
```

Alpha is 1.0, the first ridge example in chapter 23. It is the same number for the 20-column model and the subset model. Tuning it is item 3. The scaler is fit on the rows inside that fit only.

**Second model, test window only:** the same random forest as section 6.4, fit once on all training rows. It gives a second opinion on whether the subset changes the book. It is not refit along the yearly walk-forward. A 60-month forest walk-forward is a runtime choice, not a better design, and part 1 keeps the process check on the fast model.

**Fit target:** `y_fit` from section 6.1, built from the rows inside the fit.

**When the model is fit:**

- Walk-forward: before each traded snapshot, refit ridge on every training snapshot strictly earlier than that date. The feature list is the one chosen at the most recent origin, and it stays fixed for the 12 traded months.
- Test: fit once on all 120 training snapshots. Predict every test row. Refitting each test month on the same training file would reproduce the same coefficients.

**Book, each snapshot:**

- Rank that month's stocks by the prediction.
- Hold every stock whose prediction is at least the 80th percentile of predictions that month. Ties at the cutoff stay in the book.
- Equal-weight the names that are held.
- The month's strategy return is the equal-weight mean of those names' raw `R1M_Usd`.

Top quintile is about 200 names in this universe. A decile book would be a different strategy, and switching to it after seeing results would be a second experiment. One book is pre-committed.

**Equal-weight universe, reference line:** each snapshot, the mean of raw `R1M_Usd` across every stock present that month. This is the assignment's equally weighted market portfolio, because the files have no market-cap column. Part 1 plots it and reports its test-window annualized return with the formula in section 8, so later parts inherit the same definition and the same 108 months.

---

## 8. How performance is measured

Let `r` be a series of monthly portfolio returns, in snapshot order. Pandas sample standard deviation (`ddof=1`), matching chapter 15b's `.std()`.

```text
annualized_return = ((1 + r).prod()) ** (12 / len(r)) - 1
annualized_vol    = r.std(ddof=1) * sqrt(12)
sharpe            = (r.mean() / r.std(ddof=1)) * sqrt(12)
```

Chapter 15b uses 252 because its returns are daily. These series are monthly, so the exponent and the square root use 12. The Sharpe subtracts no Treasury bill. It is the textbook ratio, and it is comparable across books because every book is scaled the same way.

Equity and drawdown, as in chapter 16:

```text
equity = (1 + r).cumprod()
drawdown = equity / equity.cummax() - 1
max_drawdown = drawdown.min()
```

If any monthly portfolio return is less than or equal to −1, stop and report it. An equal-weight book of about 200 names should not hit that bound. A single name in this file does go to about −0.92, and that name alone cannot drag a 200-name book to −1.

Also report, for each book:

- Mean monthly Spearman correlation between the model's prediction and raw `R1M_Usd`.
- Fraction of months the book beats that month's equal-weight universe.
- Average one-way turnover. Weights are `1/n_held` on held names and 0 on other names. One-way turnover from month `t-1` to `t` is `0.5 * sum(|w_t - w_{t-1}|)` over the union of names. Average that quantity over the months after the first.

**Paired comparison.** `d = r_subset - r_all20` on the same months.

```text
t = d.mean() / (d.std(ddof=1) / sqrt(len(d)))
```

This treats months as independent. Also report a Newey-West t-stat with lag 3. Let `e` be `d` minus its mean, `n = len(d)`, and

```text
gamma(k) = (1/n) * sum_{t=k+1}^{n} e_t e_{t-k}
S = gamma(0) + 2 * sum_{k=1}^{3} (1 - k/4) * gamma(k)
t_nw = d.mean() / sqrt(S / n)
```

Lag 3 covers a quarter of monthly dependence. If `S` is not positive, report the textbook t-stat and say the Newey-West variance was not positive.

**Two views of the same test pass.**

1. Outlier cap. Recompute each book's monthly return after clipping every stock's raw return at 1.0 (100%) before the equal-weight average. Names held do not change. This asks whether one print of 20x or 30x decided the race. The primary numbers use unclipped returns. A flip in ranking under the cap means the result is tail-driven, which the decision rule records as no clear improvement.
2. Cost haircut. `r_net = r - 0.0010 * one_way_turnover`. Ten basis points per unit of one-way turnover. Replacing the whole book costs 10 bp that month. This is a sensitivity knob, not a spread model. Report whether the subset-versus-all-20 ranking survives it.

---

## 9. What the notebook contains, in order

The notebook path is:

`11_mlfactor_long_only_project_2026/part1_feature_selection/02_part1_feature_selection.ipynb`

Markdown cells carry the explanation. Code comments say what a block is for. The rubric grades commenting on the project notebook, and this notebook is the part 1 piece of that work.

1. **Question and rules.** Item 1 in the student's own words. The leakage rules from section 5. A one-paragraph map of the rest of the notebook.
2. **Load and checks.** Shapes, date span, names per month, missing values, duplicate keys, the uniform-percentile check (min, max, mean, std, nunique). State the return-month calendar from section 2.
3. **Glossary.** The name table. Say which names are uncertain. Show a few rows so the percentile bins are visible.
4. **Redundancy.** Annotated correlation heatmap. Dendrogram from average linkage on `1 - |corr|`. Print the clusters at distance 0.20 and 0.30. A short PCA on the standardized training features: cumulative explained variance and which tight cluster dominates the top loadings. The prose says PCA justifies the 0.80 gate and is not the submitted subset.
5. **Single-column association.** Monthly Spearman IC table with mean, std, t-stat, and hit rate. Bar chart of mean IC with an error bar of `std / sqrt(n_months)`. Prose on the size of a 0.01 IC and on the fat tail's share of sum of squares.
6. **`select_features`, taught on the full training frame.** Show one lasso coefficient path (chapter 23's picture) before the loop, so the alpha grid and the zeros are visible. Then run the folds. Show frequency, chosen alphas, and a small table of coefficients.
7. **Permutation and the scorecard.** Explain a within-month shuffle with one column as the example. Show `perm_drop` beside `xgb_gain`. Build the score. Show who was eligible, who was skipped for correlation, and who was cut by the cap. Print `subset_final` and the fallback flag.
8. **Yearly process check.** The five origins. Ridge walk-forward of all 20 columns against that year's subset. Equity curves on one chart with the equal-weight universe. The metric table. This section's prose says this path never uses the test file, and that each subset was chosen on earlier snapshots only.
9. **Write `subset_final.csv` and the scorecard.** Do this before section 10.
10. **Test backtest, one pass.** Fit ridge and the forest on all training rows. Predict the 108 test snapshots. Table from section 8 for the equal-weight universe, ridge-all, ridge-subset, forest-all, forest-subset. Equity curves and drawdown curves. Paired t-stats. Capped-return view. Turnover haircut.
11. **Decision.** Fill the sentence from section 1 using the tables. Name the economic ideas that survived (one cash-flow column, one yield, one leverage measure, and so on) and the duplicates that were dropped. The presentation guideline for the course says to describe the analysis that led to the features. This cell is that description.
12. **Save** `strategy_comparison.csv` and `yearly_walkforward.csv` beside the notebook.

Figures worth making, and what the reader is supposed to see:

- Heatmap: the turnover block and the leverage block are dark, the yield columns are not.
- IC bars: every bar is small relative to its error bar.
- Lasso path: correlated columns hand the coefficient to one survivor as alpha rises.
- Equity curves: three lines over 2010–2018, with the drawdown panel underneath, in the spirit of chapter 16.

---

## 10. Files the notebook writes

All of them live in `part1_feature_selection/`.

| File | Contents |
| --- | --- |
| `subset_final.csv` | One row per chosen feature: name, score, lasso frequency, perm drop, mean IC, t-stat, and the reason it was kept. |
| `feature_scorecard.csv` | All 20 features, every scorecard column, eligibility, and whether the greedy pass kept it. |
| `yearly_walkforward.csv` | One row per traded training snapshot: date, origin, subset used, ridge return all-20, ridge return subset, equal-weight return. |
| `strategy_comparison.csv` | One row per test strategy and per view (raw, capped, cost-haircut): the section 8 metrics. |

`03_learnings_from_implementation.md` is updated from these files after the run. It records deviations, the fallback flag, the overlap of yearly subsets with `subset_final`, and the decision sentence.

---

## 11. How a later review can beat this

A better part 1 is a better long-only result under the same trading rule, or the same result from a simpler and more stable subset, with the test labels still unused during selection.

Legitimate knobs, each changed on its own so the comparison with this plan is readable:

- The score weights `0.40 / 0.40 / 0.20`.
- The correlation gate `0.80`, or keeping two columns from a cluster when their correlation is high and their lasso signs differ.
- The eligibility gates.
- Choosing the lasso alpha by the MSE side column instead of by Spearman correlation.
- A rank-transformed `y_fit` (within-month percentile of the return) in place of the clipped demeaned return.
- Top-quintile versus a fixed name count, pre-committed before the test pass.

The current plan is the baseline those attempts have to beat on the section 8 test table, together with the walk-forward agreement required by the decision rule.

---

## 12. What this design is ready to be wrong about

- The signals may be so small that `subset_final` and the all-20 book differ by noise. The decision rule has a "similar performance" outcome for that case. The notebook reports it when that is the result.
- The crisis years sit inside the training file. They affect both the full-sample lasso and the later walk-forward origins. That is the sample the prompt gives us for fitting.
- `Bb_Yld` and `Netdebtyield` may be misread in the glossary. Misreading a name changes the paragraph in the decision cell. It does not change the score.
- Ridge with alpha 1 may be a poor model for both column sets. Part 1 still uses it, because the question is the column set. A later part can change the model.
- The first walk-forward origin has a single lasso fold. Frequency on that origin is coarse.
