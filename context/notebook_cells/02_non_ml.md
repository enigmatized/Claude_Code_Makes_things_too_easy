MARKDOWN

## Two non-ML books

This section is the two non-ML books on the official test: equal weight, and the positive-volatility momentum book. Full baskets are the other constructed book on the same months; risk and reward are both in the table, gross of costs, and cash earns nothing.

CODE

# Newest final-book save, official test months only. Non-ML books from STORY. This is the graded comparison against equal weight.
from research.notebook_views import non_ml_figure, non_ml_table, source_note, style_table
print(source_note("final"))
display(style_table(non_ml_table("test")))
display(non_ml_figure("test"))
