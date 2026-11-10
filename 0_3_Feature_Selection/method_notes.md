**All 20 factors** is the exploratory reference. Each learned recipe selects
at most ten inputs using only outcomes available at that historical training
date, then independently retrains each classifier. Preprocessing and selection
remain inside the model pipeline, including inside future tuning folds.

- **ANOVA top 10:** rank factors individually by training separation between
  loss and non-loss outcomes; retain the highest-ranked varying inputs.
- **Correlation-pruned up to 10:** follow the ANOVA ranking while rejecting
  factors too correlated with a higher-ranked retained input. Use the saved
  correlation threshold; do not backfill rejected duplicates.
- **Random forest top 10:** fit a separate selector forest on all candidate
  inputs and rank their mean decrease in impurity. Retain the highest-ranked
  varying inputs, then fit the chosen prediction model. A forest selector can
  supply inputs to logistic regression, naive Bayes, boosting or another forest.

**Reading the forest chart:** longer bars indicate more training split
importance. Highlighted bars are the selected inputs. Error bars show standard
deviation across trees, not confidence intervals or future-return uncertainty.
Importance does not give the direction of an effect or establish causality.
Correlated factors can share importance, and inputs with more possible split
points can receive inflated importance. ANOVA scores and forest importances have
different units; compare selected names and later outcomes across methods.

**Reading the comparisons:** lower Brier score and log loss are better. Compare
each selected recipe with the same predictor using all twenty factors and with
the historical event-frequency control. Then compare the resulting portfolios
over identical months. Feature selection changes ML inputs; non-ML rules remain
unchanged references. A favorable return does not establish useful prediction.
Annual masks may differ; their union is not a fixed final ten-factor model.

Method references: textbook chapter 29 introduces tree importance and random
forests; [sklearn's forest example](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)
illustrates importance and tree variability. See its
[permutation-importance guide](https://scikit-learn.org/stable/modules/permutation_importance.html)
for further interpretation limits. Permutation selection is not part of this study.
