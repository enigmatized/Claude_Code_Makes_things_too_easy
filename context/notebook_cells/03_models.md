MARKDOWN
## Four model types

Four classifier families are on the same books and the same official-test months: logistic regression, the forest, and histogram gradient boosting have tunable hyperparameters, and Gaussian naive Bayes is the fourth family. This table is CAGR, and the Sharpe table is the risk companion. These official-test books were fit with factory settings; the search itself is a later cell.

CODE
# Newest final-book save, official test. Every saved classifier on the story books. CAGR, then Sharpe. Factory settings, not the grid search.
from research.notebook_views import model_sharpe_table, model_table, source_note
print(source_note("final"))
display(model_table("test").style.format({column: "{:.2%}" for column in model_table("test").columns if column != "Book"}, na_rep="—").hide(axis="index"))
display(model_sharpe_table("test").style.format({column: "{:.3f}" for column in model_sharpe_table("test").columns if column != "Book"}, na_rep="—").hide(axis="index"))
