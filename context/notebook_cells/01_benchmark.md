MARKDOWN
## Equal-weight test return

This cell is the equal-weight market portfolio on the official test only. Each month is the average of simple returns, then the compound annualized return.

CODE
# Independent equal-weight file, official test only (January 2010–December 2018). Each month is the mean of simple returns, then ((1+r).prod())**(12/n)-1. This is the graded annualized return.
from research.notebook_views import equal_weight_benchmark, equal_weight_sentence, source_note, style_table
print(source_note("benchmark"))
display(style_table(equal_weight_benchmark().drop(columns=["Source"], errors="ignore")))
print(equal_weight_sentence())
