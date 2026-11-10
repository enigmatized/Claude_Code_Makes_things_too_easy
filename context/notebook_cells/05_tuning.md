## What GridSearchCV searched

Hyperparameter search uses sklearn GridSearchCV on training data only, scored with a probability metric (log loss), not with portfolio return. The holdout months are January 2007–December 2009. The question for the grade is whether the searched settings change the book; the official test later in the notebook was not refit with these settings.

```python
# Newest saved GridSearchCV holdout. Training years only. Search table, then whether the holdout book moved. Not the official test.
from research.notebook_views import source_note, tuning_note, tuning_portfolio_table, tuning_search_table
print(source_note("tuning"))
display(tuning_search_table().style.format({"Best score": "{:.4f}"}, na_rep="—").hide(axis="index"))
display(tuning_portfolio_table().style.format({column: "{:.2%}" for column in tuning_portfolio_table().columns if column not in {"Strategy", "Model"}}, na_rep="—").hide(axis="index"))
print(tuning_note())
```
