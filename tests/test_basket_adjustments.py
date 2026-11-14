"""Directional basket replay: cash budgets, saved provenance, no new fitting."""
import os
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from test_basket_momentum import panel
from research.basket_adjustments import (
    ADJUSTED_VERSIONS, BASKET_ADJUSTED_VERSIONS, BASKET_DOWNSIDE_VERSION, BASKET_UPSIDE_VERSION,
    DOWNSIDE_VERSION, UPSIDE_VERSION, create_adjusted_basket_study, _read_forecasts,
)
from research.basket_features import load_feature_book_scores, save_feature_book_scores, score_feature_books
from research.basket_momentum import TOP_VOL_VERSION, create_basket_study, load_basket_study, save_basket_study
from research.basket_momentum import _digest
from research.legacy import KEYS
from research.runner import ExperimentRunner, RunConfig
from research.store import ExperimentStore


class DirectionalBasketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        history = panel(months=30, stocks=8)
        history.to_csv(cls.root / "mlfactor_long_only_train.csv", index=False)
        cls.store = ExperimentStore(project_root=cls.root)
        cls.base = create_basket_study(cls.root, n_clusters=2, validation_history_months=24)
        cls.directory = save_basket_study(cls.base, cls.store.root / "basket_studies")
        cls.runs = {}
        for target in ("loss", "upside"):
            config = RunConfig(("equal_weight_universe",), include_ml=True, validation_history_months=24,
                               prediction_target=target)
            cls.runs[target] = cls.store.save(ExperimentRunner(cls.root).run(config))

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def replay(self, **kwargs):
        with patch("research.models._fit_slice", side_effect=AssertionError("Must reuse saved fits")):
            return create_adjusted_basket_study(self.store, self.directory, self.runs["loss"],
                                                 self.runs["upside"], **kwargs)

    def test_actual_probabilities_control_weights_and_cash_without_changing_membership(self):
        result = self.replay()
        self.assertEqual(result["metadata"]["new_prediction_fits"], 0)
        for frame in ("monthly", "metrics", "baskets", "assignments"):
            original = result[frame].loc[result[frame].model_key.eq("none")].drop(columns="model_key")
            # New audit columns do not change a single original value.
            pd.testing.assert_frame_equal(original[self.base[frame].columns].reset_index(drop=True),
                                          self.base[frame].reset_index(drop=True), check_dtype=False)
        held = result["assignments"]
        down = held.loc[held.version.eq(DOWNSIDE_VERSION)]
        up = held.loc[held.version.eq(UPSIDE_VERSION)]
        np.testing.assert_allclose(down.weight, down.baseline_weight * (1 - .5 * down.probability_of_loss))
        raw = up.baseline_weight * (1 + .5 * up.probability_of_up.ge(.6))
        groupers = [up.date, up.momentum_months]
        budget = up.baseline_weight.groupby(groupers).transform("sum")
        total = raw.groupby(groupers).transform("sum")
        np.testing.assert_allclose(up.weight, raw * budget / total.where(total.gt(0), 1))
        adjusted = held.loc[held.version.isin(ADJUSTED_VERSIONS)]
        self.assertTrue(adjusted.loc[adjusted.baseline_weight.eq(0), "weight"].eq(0).all())
        self.assertTrue(adjusted.loc[adjusted.weight.gt(0), "trailing_momentum"].gt(0).all())
        self.assertTrue(adjusted.loc[adjusted.weight.gt(0), "basket_momentum"].gt(0).all())
        self.assertTrue(adjusted.groupby(["date", "version", "momentum_months"]).weight.sum().le(1+1e-10).all())
        monthly = result["monthly"]
        original = monthly.loc[monthly.version.eq(TOP_VOL_VERSION)].sort_values(["momentum_months", "return_month"])
        down_cash = monthly.loc[monthly.version.eq(DOWNSIDE_VERSION)].sort_values(["momentum_months", "return_month"])
        up_cash = monthly.loc[monthly.version.eq(UPSIDE_VERSION)].sort_values(["momentum_months", "return_month"])
        self.assertTrue((down_cash.cash_weight.to_numpy() >= original.cash_weight.to_numpy()-1e-10).all())
        np.testing.assert_allclose(up_cash.cash_weight, original.cash_weight, atol=1e-10)
        self.assertTrue(set(BASKET_ADJUSTED_VERSIONS).issubset(result["metadata"]["versions"]))
        basket_down = held.loc[held.version.eq(BASKET_DOWNSIDE_VERSION)]
        basket_up = held.loc[held.version.eq(BASKET_UPSIDE_VERSION)]
        self.assertFalse(basket_down.empty)
        np.testing.assert_allclose(
            basket_down.weight, basket_down.baseline_weight * (1 - .5 * basket_down.probability_of_loss))
        raw = basket_up.baseline_weight * (1 + .5 * basket_up.probability_of_up.ge(.6))
        groupers = [basket_up.date, basket_up.momentum_months, basket_up.model_key]
        budget = basket_up.baseline_weight.groupby(groupers).transform("sum")
        total = raw.groupby(groupers).transform("sum")
        np.testing.assert_allclose(basket_up.weight, raw * budget / total.where(total.gt(0), 1))
        self.assertTrue(basket_down.loc[basket_down.baseline_weight.eq(0), "weight"].eq(0).all())
        self.assertTrue(basket_up.loc[basket_up.baseline_weight.eq(0), "weight"].eq(0).all())
        self.assertTrue(basket_down.loc[basket_down.weight.gt(0), "trailing_momentum"].gt(0).all())
        self.assertTrue(basket_down.loc[basket_down.weight.gt(0), "basket_momentum"].gt(0).all())
        source = monthly.loc[monthly.version.eq("Basket momentum + cash") & monthly.model_key.eq("none")]
        source = source.sort_values(["momentum_months", "return_month"])
        for key in result["metadata"]["model_keys"]:
            full_down = monthly.loc[monthly.version.eq(BASKET_DOWNSIDE_VERSION) & monthly.model_key.eq(key)]
            full_up = monthly.loc[monthly.version.eq(BASKET_UPSIDE_VERSION) & monthly.model_key.eq(key)]
            full_down = full_down.sort_values(["momentum_months", "return_month"])
            full_up = full_up.sort_values(["momentum_months", "return_month"])
            self.assertTrue((full_down.cash_weight.to_numpy() >= source.cash_weight.to_numpy() - 1e-10).all())
            np.testing.assert_allclose(full_up.cash_weight, source.cash_weight, atol=1e-10)

    def test_zero_strength_roundtrip_and_saved_only_ui(self):
        result = self.replay(strength=0)
        for version in (*ADJUSTED_VERSIONS, *BASKET_ADJUSTED_VERSIONS):
            rows = result["assignments"].loc[result["assignments"].version.eq(version)]
            np.testing.assert_array_equal(rows.weight, rows.baseline_weight)
        directory = save_basket_study(result, self.store.root / "basket_studies")
        loaded = load_basket_study(directory)
        self.assertIn("predictive", loaded)
        self.assertIn("fits", loaded)
        script = ("from basket_ui import render_basket_strategy\n"
                  "from research.store import ExperimentStore\n"
                  f"render_basket_strategy(ExperimentStore(project_root={str(self.root)!r}))")
        with patch.dict(os.environ, {"MLFACTOR_READ_ONLY": "1"}):
            app = AppTest.from_string(script).run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key="basket_prediction_model").value, "Logistic regression")
            self.assertEqual([tab.label for tab in app.tabs], ["K-means", "Hierarchical clustering", "Adding a direction", "Momentum acceleration"])
            app.checkbox(key="basket_overlay_inspect").check().run()
            app.selectbox(key="basket_screen_1").select(DOWNSIDE_VERSION).run()
            self.assertFalse(app.exception)
            app.selectbox(key="basket_selection_basket").select("K-means full groups").run()
            app.selectbox(key="basket_screen_1").select(BASKET_DOWNSIDE_VERSION).run()
            self.assertFalse(app.exception)

    def test_feature_book_scores_reuse_forecasts_without_allocation_files(self):
        with patch("research.models._fit_slice", side_effect=AssertionError("Must reuse saved fits")):
            light = create_adjusted_basket_study(
                self.store, self.directory, self.runs["loss"], self.runs["upside"], keep_allocations=False)
        self.assertNotIn("assignments", light)
        self.assertTrue(light["metrics"].version.eq(BASKET_DOWNSIDE_VERSION).any())
        self.assertEqual(light["metadata"]["new_prediction_fits"], 0)
        recipes = [
            {"key": "all_20", "label": "All 20", "loss_run_id": self.runs["loss"], "upside_run_id": self.runs["upside"]},
            {"key": "anova_10", "label": "ANOVA", "loss_run_id": self.runs["loss"], "upside_run_id": self.runs["upside"]},
        ]
        with patch("research.models._fit_slice", side_effect=AssertionError("Must reuse saved fits")):
            scored = score_feature_books(self.store, self.directory, recipes)
        self.assertTrue(scored["metadata"]["test_csv_loaded"] is False)
        self.assertEqual(set(scored["metrics"].loc[scored["metrics"].recipe.eq(""), "version"]),
                         {"Equal weight", "Top 10 positive-vol baskets", "Basket momentum + cash"})
        self.assertEqual(scored["metrics"].loc[scored["metrics"].recipe.ne(""), "recipe"].nunique(), 2)
        directory = save_feature_book_scores(scored, self.store.root)
        loaded = load_feature_book_scores(directory)
        self.assertEqual(len(loaded["metrics"]), len(scored["metrics"]))

    def test_rejects_wrong_target_changed_source_and_rendering_replay(self):
        with self.assertRaisesRegex(ValueError, "requested target"):
            create_adjusted_basket_study(self.store, self.directory, self.runs["upside"], self.runs["loss"])
        path = self.root / "mlfactor_long_only_train.csv"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "same training file"):
                self.replay()
        finally:
            path.write_bytes(original)
        with patch.dict(os.environ, {"MLFACTOR_READ_ONLY": "1"}):
            with self.assertRaises(RuntimeError):
                self.replay()

    def test_forecast_validation_rejects_future_fits_and_missing_stock_months(self):
        record, directory, metadata = self.store._record(self.runs["loss"])
        source = metadata["frame_paths"]["prediction_batches"]["logistic"]["predictions"]
        predictions = pd.read_parquet(directory / source)
        for failure in ("future", "missing"):
            bad = predictions.copy()
            if failure == "future":
                bad.loc[bad.index[0], "fit_snapshot"] = bad.date.iloc[0] + pd.offsets.MonthEnd(1)
            else:
                bad = bad.iloc[1:]
            path = self.root / "bad_forecasts.parquet"
            bad.to_parquet(path)
            modified = copy.deepcopy(metadata)
            modified["frame_paths"]["prediction_batches"]["logistic"]["predictions"] = str(path)
            with patch.object(self.store, "_record", return_value=(record, directory, modified)):
                with self.assertRaisesRegex(ValueError, "future training|cover exactly"):
                    _read_forecasts(self.store, self.runs["loss"], "loss", predictions[KEYS],
                                    _digest(self.root / "mlfactor_long_only_train.csv"))
if __name__ == "__main__":
    unittest.main()
