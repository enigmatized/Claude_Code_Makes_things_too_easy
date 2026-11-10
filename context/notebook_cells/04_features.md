# Did a ten-factor cut change the book?

Feature choice is judged on the training window, before the official test. The question is whether a subset of at most ten factors changes the book versus all twenty factors. The method names come from the saved recipes (ANOVA, correlation pruning, and forest importance), and the name list is the frozen official-test fit, which can differ from those training-window recipes.

```python
# Newest feature-score save, training years only. Overlay books at DISPLAY_MODEL. Did a ten-factor cut move CAGR versus all 20 factors?
from research.notebook_views import feature_cut_table, feature_name_note, source_note
print(source_note("features"))
display(feature_cut_table().style.format({column: "{:.2%}" for column in feature_cut_table().columns if column != "Book"}, na_rep="—").hide(axis="index"))
print(feature_name_note())
```
