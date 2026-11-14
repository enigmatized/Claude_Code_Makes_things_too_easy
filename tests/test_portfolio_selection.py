"""Saved-case choices, unavailable combinations and final protocol provenance."""

import json
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from portfolio_selection import (
    REGISTRY_PARTS,
    book_cases,
    grouped_cases,
    matching_cases,
    registry_cases,
)
from research.strategies import default_registry


class SavedCaseSelectionTests(unittest.TestCase):
    def test_registry_bundles_keep_every_original_key_without_inventing_pairings(self):
        cases = registry_cases(default_registry(), ("logistic",), ("loss",))
        self.assertEqual(set(cases.strategy), set(REGISTRY_PARTS))
        self.assertFalse(
            cases.duplicated(["basket", "allocation", "model_key", "target"]).any()
        )
        high = matching_cases(cases, basket="Highest trailing-volatility stocks")
        self.assertEqual(high.strategy.tolist(), ["high_volatility"])
        self.assertTrue(matching_cases(high, allocation="Momentum weights").empty)

    def test_book_mapping_preserves_recipe_fit_scope_and_model_identity(self):
        source = pd.DataFrame(
            [
                dict(
                    book="Ward top 10",
                    recipe="anova_10",
                    model_key="logistic",
                    target="loss",
                    fit_origin="2009-12-31",
                    status="Tuned",
                )
            ]
        )
        cases = book_cases(source)
        pd.testing.assert_frame_equal(cases[source.columns], source)
        self.assertEqual(cases.basket.iloc[0], "Ward top 10 positive-vol per group")
        self.assertEqual(cases.allocation.iloc[0], "1M positive momentum + cash")

    def test_grouped_cases_keep_recorded_directions_and_actual_horizons(self):
        source = pd.DataFrame(
            [
                dict(
                    version="Top 10 positive-vol baskets",
                    momentum_months=2,
                    model_key="none",
                ),
                dict(
                    version="Top 10 + downside to cash",
                    momentum_months=2,
                    model_key="logistic",
                ),
            ]
        )
        cases = grouped_cases(source, "ward")
        self.assertEqual(cases.target.tolist(), ["none", "loss"])
        self.assertEqual(
            cases.allocation.unique().tolist(), ["2M positive momentum + cash"]
        )
        self.assertTrue(cases.basket.str.startswith("Ward").all())
        self.assertTrue(matching_cases(cases, target="upside").empty)

    def test_four_control_order_and_stale_downstream_values_reset_to_available_cases(
        self,
    ):
        script = """
import pandas as pd
import streamlit as st
from portfolio_selection_ui import select_portfolio_case
cases=pd.DataFrame([
 dict(basket='A',model_key='logistic',allocation='Equal weight',target='loss',source='fit1'),
 dict(basket='A',model_key='logistic',allocation='Equal weight',target='upside',source='fit2'),
 dict(basket='B',model_key='naive_bayes',allocation='Momentum + cash',target='upside',source='fit3')])
selected=select_portfolio_case(cases,'pilot')
st.dataframe(selected)
"""
        with patch("pandas.read_csv", side_effect=AssertionError("No raw data")), patch(
            "research.runner.ExperimentRunner.run",
            side_effect=AssertionError("No replay"),
        ):
            app = AppTest.from_string(script).run()
            self.assertFalse(app.exception)
            self.assertEqual(
                [widget.label for widget in app.selectbox],
                [
                    "1. Asset basket",
                    "2. Prediction model",
                    "3. Allocation strategy",
                    "4. Prediction target",
                ],
            )
            app.selectbox(key="pilot_target").select("Next up").run()
            self.assertEqual(app.dataframe[0].value.source.tolist(), ["fit2"])
            app.selectbox(key="pilot_target").select("Next down").run()
            app.selectbox(key="pilot_basket").select("B").run()
            self.assertFalse(app.exception)
            self.assertEqual(
                app.selectbox(key="pilot_model").options, ["Gaussian naive Bayes"]
            )
            self.assertEqual(
                app.selectbox(key="pilot_allocation").value, "Momentum + cash"
            )
            self.assertEqual(app.selectbox(key="pilot_target").options, ["Next up"])
            self.assertEqual(app.dataframe[0].value.source.tolist(), ["fit3"])

    def test_nonpredictive_view_has_only_basket_and_allocation_controls(self):
        script = """
from portfolio_selection import registry_cases
from portfolio_selection_ui import select_portfolio_case
select_portfolio_case(registry_cases(['equal_weight_universe']),'baseline',predictive=False)
"""
        app = AppTest.from_string(script).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.selectbox), 2)


class FinalComparisonTests(unittest.TestCase):
    def test_comparison_keeps_source_protocol_values_and_rejects_mismatched_calendars(
        self,
    ):
        script = """
import pandas as pd
from final_comparison_ui import offer_final_comparison,render_final_comparison
metrics=pd.DataFrame([dict(variant='Original',cagr=.123,sharpe=.6,max_drawdown=-.2)])
for prefix,period in [('one','2010-01-31'),('two','2010-02-28')]:
 monthly=pd.DataFrame(dict(variant=['Original'],return_month=[pd.Timestamp(period)],
    snapshot_date=[pd.Timestamp(period)-pd.offsets.MonthEnd(1)],simple_return=[.01],wealth=[1.01],drawdown=[0.],cash_weight=[.5]))
 offer_final_comparison(metrics,monthly,protocol=prefix,source='frozen_'+prefix,
    case_fields=['variant'],prefix=prefix,window='test')
render_final_comparison()
"""
        app = AppTest.from_string(script).run()
        app.button(key="one_add_comparison").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.dataframe[0].value.CAGR.tolist(), [0.123])
        chart = json.loads(app.get('plotly_chart')[0].proto.figure.spec)
        self.assertEqual(len(chart['data']), 3)
        self.assertEqual(len({trace['line']['color'] for trace in chart['data']}), 1)
        app.button(key="two_add_comparison").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(set(app.dataframe[0].value.Protocol), {"one", "two"})
        self.assertTrue(
            any("different evaluation months" in message.value for message in app.info)
        )
        self.assertFalse(app.get("plotly_chart"))


if __name__ == "__main__":
    unittest.main()
