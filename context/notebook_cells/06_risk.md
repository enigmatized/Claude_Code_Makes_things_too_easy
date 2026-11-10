MARKDOWN

## Risk and reward

Risk and reward are shown together for the same books. When the training save exists, the table has three windows: the official test, the training window, and the path that compounds the training months and then the test months. The line on the chart is the first official-test month; returns are gross of costs, and cash earns nothing.

CODE

# Three windows from the newest final-book save and the saved training months. No new fit. The chart compounds training into the test when both exist.
from research.notebook_views import path_figure, risk_note, risk_table, style_table
display(style_table(risk_table()))
display(path_figure())
print(risk_note())
