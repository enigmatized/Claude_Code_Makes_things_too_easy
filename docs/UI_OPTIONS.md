# Python UI options — 2026-11-08

Recommendation: keep the working Streamlit viewer through the saved-results and
submission work. For a subsequent custom visual redesign, prototype NiceGUI
against the same artifact-reading API. The research engine and report do not
need to depend on the UI framework.

| Option | Fit for this project | Cost/trade-off |
| --- | --- | --- |
| Streamlit | Existing tested controls, plots and research workflow. Best near-term delivery path. | More constrained application layouts and interaction model. Cache-first storage is now independent of its rerun mechanism. |
| NiceGUI | Preferred candidate for a more custom Python application: flexible components, Quasar/Tailwind styling, explicit events and Plotly support. | New dependency and a UI rewrite; first prove one saved-results comparison page. |
| Panel | Strong analytical-dashboard alternative; notebook/plot integration and Material/Fast templates. Already installed locally. | Different reactive model and component/layout APIs to learn. |
| Reflex | Python-authored full-stack UI using components compiled to browser JavaScript. | Adds a frontend build/state architecture beyond what this local research viewer currently needs. |

“Professional” should first mean consistent result identities, legible comparisons,
clear explanations, responsive saved-result loading and reproducible documents.
A custom interface can then improve visual polish without changing calculations.

The suggested redesign prototype is a saved-run library with one comparison
screen: strategy/model/recipe chips, return/risk cards, paired curves, allocation
timeline and a visible provenance drawer. Compare it with the current app before
committing to a migration. No UI framework migration was made in this update.

Primary sources checked:

- [NiceGUI repository and feature overview](https://github.com/zauberzeug/nicegui).
- [Panel Material template](https://panel.holoviz.org/reference/templates/Material.html).
- [Reflex introduction](https://reflex.dev/docs/getting-started/introduction/).
- [Streamlit caching model](https://docs.streamlit.io/develop/concepts/architecture/caching).
