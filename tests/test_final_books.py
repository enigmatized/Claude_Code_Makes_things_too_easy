"""The official four-book pass fits once and the page can show that file."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.basket_adjustments import BASKET_DOWNSIDE_VERSION, DOWNSIDE_VERSION
from research.basket_momentum import TOP_VOL_VERSION
from research.basket_momentum import _digest
from research.final_books import (
    list_final_books, load_final_books, matching_training_monthly, save_final_books,
    score_final_books, window_books,
)
from research.legacy import KEYS, TARGET
from test_research import toy_panel


def _write_split(root, months=90, train_months=72):
    panel = toy_panel(months, 8)
    dates = panel.date.drop_duplicates().sort_values()
    panel.loc[panel.date.isin(dates.iloc[:train_months])].to_csv(root / "mlfactor_long_only_train.csv", index=False)
    panel.loc[panel.date.isin(dates.iloc[train_months:])].to_csv(root / "mlfactor_long_only_test.csv", index=False)
    return panel


def _fake_study(root):
    dates = pd.to_datetime(["2009-12-31", "2010-01-31"])
    returns = pd.to_datetime(["2010-01-31", "2010-02-28"])
    rows = []
    monthly = []
    versions = [
        ("Equal weight", 0, "none"),
        (TOP_VOL_VERSION, 1, "none"),
        (DOWNSIDE_VERSION, 1, "logistic"),
        (BASKET_DOWNSIDE_VERSION, 1, "logistic"),
    ]
    for version, horizon, model in versions:
        rows.append({
            "version": version, "momentum_months": horizon, "model_key": model,
            "cagr": 0.13, "annualized_volatility": 0.16, "sharpe": 0.87,
            "max_drawdown": -0.2, "avg_cash_weight": 0.1, "avg_names": 25,
        })
        for snapshot, earned, wealth in zip(dates, returns, (1.01, 1.02)):
            monthly.append({
                "snapshot_date": snapshot, "return_month": earned, "simple_return": 0.01,
                "wealth": wealth, "drawdown": 0.0, "turnover_one_way": 0.0, "n_held": 10,
                "version": version, "momentum_months": horizon, "cash_weight": 0.1,
                "equity_weight": 0.9, "model_key": model,
            })
    result = {
        "metrics": pd.DataFrame(rows),
        "monthly": pd.DataFrame(monthly),
        "fits": pd.DataFrame([{
            "model_key": "logistic", "prediction_target": "loss", "n_features": 10,
            "fitted_features": "[\"Div_Yld\"]", "fit_snapshot": dates[0],
            "trained_through_return_month": dates[0], "training_rows": 8,
            "fit_policy": "frozen_at_origin",
        }]),
        "metadata": {
            "protocol": "final_four_books", "test_csv_loaded": True, "recipe": ["Keep this recipe."],
            "fit_policy": "frozen_at_origin", "fit_snapshot": "2009-12-31",
            "trained_through_return_month": "2009-12-31", "n_features": 10,
            "adjustment_strength": 0.5, "upside_threshold": 0.6,
            "first_return_month": "2010-01-31", "last_return_month": "2010-02-28", "months": 2,
            "fitted_features": {"logistic": {"loss": ["Div_Yld"], "upside": ["Div_Yld"]}},
        },
    }
    return save_final_books(result, root)


def _monthly_block(start, returns, version="Equal weight", model="none", horizon=0):
    dates = pd.date_range(start, periods=len(returns), freq="ME")
    return pd.DataFrame({
        "snapshot_date": dates - pd.offsets.MonthEnd(1),
        "return_month": dates,
        "simple_return": returns,
        "wealth": 1.0,
        "drawdown": 0.0,
        "turnover_one_way": [float("nan"), *([0.1] * (len(returns) - 1))],
        "n_held": 2,
        "version": version,
        "momentum_months": horizon,
        "cash_weight": 0.0,
        "equity_weight": 1.0,
        "model_key": model,
    })


class FinalBookTests(unittest.TestCase):
    def test_entire_path_compounds_training_then_the_official_test(self):
        training = _monthly_block("2009-11-30", [0.10, 0.10])
        test = pd.concat([
            _monthly_block("2010-01-31", [0.0, 0.0]),
            _monthly_block("2010-01-31", [0.05, 0.05], version="Top 10 positive-vol baskets", horizon=1),
        ], ignore_index=True)
        with patch("research.final_books.load_panel", side_effect=AssertionError("opened a csv")):
            windows = window_books(test, training)
        self.assertEqual(set(windows), {"test", "training", "entire"})
        self.assertEqual(windows["test"][1].return_month.min(), pd.Timestamp("2010-01-31"))
        self.assertEqual(windows["test"][1].return_month.max(), pd.Timestamp("2010-02-28"))
        entire = windows["entire"][1].sort_values("return_month")
        equal = entire.loc[entire.version.eq("Equal weight")]
        self.assertEqual(list(equal.return_month), list(pd.to_datetime(
            ["2009-11-30", "2009-12-31", "2010-01-31", "2010-02-28"])))
        self.assertAlmostEqual(equal.wealth.iloc[-1], 1.21)
        self.assertAlmostEqual(windows["test"][1].loc[lambda frame: frame.version.eq("Equal weight"), "wealth"].iloc[-1], 1.0)
        self.assertNotIn("Top 10 positive-vol baskets", set(windows["entire"][0].version))
        self.assertIn("Top 10 positive-vol baskets", set(windows["test"][0].version))
        with self.assertRaisesRegex(ValueError, "must meet"):
            window_books(_monthly_block("2010-03-31", [0.0, 0.0]), training)

    def test_saved_training_months_are_found_without_opening_a_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frame = _monthly_block("2009-11-30", [0.01, 0.02])
            directory = root / "basket_studies" / "baskets_example"
            directory.mkdir(parents=True)
            path = directory / "monthly.parquet"
            frame.to_parquet(path, index=False)
            metadata = {
                "protocol": "correlation_basket_momentum",
                "directional_adjustment": True,
                "versions": ["Baskets + downside to cash"],
                "first_return_month": "2009-11-30",
                "last_return_month": "2009-12-31",
                "frame_sha256": {"monthly": _digest(path)},
            }
            (directory / "study.json").write_text(json.dumps(metadata))
            with patch("research.final_books.load_panel", side_effect=AssertionError("opened a csv")):
                loaded, meta = matching_training_monthly(root)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(meta["first_return_month"], "2009-11-30")

    def test_window_control_can_show_the_entire_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _fake_study(root)
            extra = []
            for version, horizon, model in (
                ("Equal weight", 0, "none"),
                (TOP_VOL_VERSION, 1, "none"),
                (DOWNSIDE_VERSION, 1, "logistic"),
                (BASKET_DOWNSIDE_VERSION, 1, "logistic"),
            ):
                extra.append(_monthly_block("2009-11-30", [0.10, 0.10], version, model, horizon))
            training = pd.concat(extra, ignore_index=True)
            directory = root / "basket_studies" / "baskets_example"
            directory.mkdir(parents=True)
            path = directory / "monthly.parquet"
            training.to_parquet(path, index=False)
            (directory / "study.json").write_text(json.dumps({
                "protocol": "correlation_basket_momentum",
                "directional_adjustment": True,
                "versions": ["Baskets + downside to cash"],
                "first_return_month": "2009-11-30",
                "last_return_month": "2009-12-31",
                "frame_sha256": {"monthly": _digest(path)},
            }))
            app = self._render(root)
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key="final_books_window").value, "Official test only")
            self.assertTrue(any("Official test only" in item.value for item in app.caption))
            app.selectbox(key="final_books_window").select("Entire path").run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Entire path" in item.value for item in app.caption))
            self.assertTrue(any("Book" in frame.value for frame in app.dataframe))

    def test_read_only_refuses_before_opening_a_file(self):
        with tempfile.TemporaryDirectory() as folder, patch(
                "research.final_books.load_panel", side_effect=AssertionError("opened a csv")):
            with patch.dict(os.environ, {"MLFACTOR_READ_ONLY": "1"}):
                with self.assertRaises(RuntimeError):
                    score_final_books(folder)

    def test_frozen_fit_ignores_test_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            panel = _write_split(root)
            dates = panel.date.drop_duplicates().sort_values()
            train_rows = int(panel.date.isin(dates.iloc[:72]).sum())
            with patch("research.store.ExperimentStore.save", side_effect=AssertionError("not a catalog run")):
                first = score_final_books(root, models=("logistic",))
            self.assertEqual(first["metadata"]["training_rows"], train_rows)
            self.assertEqual(first["metadata"]["fit_policy"], "frozen_at_origin")
            self.assertEqual(first["metadata"]["n_features"], 10)
            self.assertTrue(first["metadata"]["test_csv_loaded"])
            self.assertFalse(first["metadata"]["official_equal_weight_check"])
            self.assertEqual(pd.Timestamp(first["metadata"]["fit_snapshot"]), dates.iloc[72])
            self.assertLessEqual(
                pd.Timestamp(first["metadata"]["trained_through_return_month"]), dates.iloc[72])
            self.assertEqual(first["fits"].fit_policy.nunique(), 1)
            self.assertTrue(first["metrics"].version.eq(DOWNSIDE_VERSION).any())
            self.assertTrue(first["metrics"].version.eq(BASKET_DOWNSIDE_VERSION).any())
            test = panel.loc[panel.date.isin(dates.iloc[72:])].copy()
            test[TARGET] = 0.2
            test.to_csv(root / "mlfactor_long_only_test.csv", index=False)
            second = score_final_books(root, models=("logistic",))
            for target, column in (("loss", "probability_of_loss"), ("upside", "probability_of_up")):
                left = first["predictions"]["logistic"][target][KEYS + [column]].sort_values(KEYS)
                right = second["predictions"]["logistic"][target][KEYS + [column]].sort_values(KEYS)
                pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True))
            self.assertEqual(first["metadata"]["fitted_features"], second["metadata"]["fitted_features"])
            ew = lambda result: result["metrics"].loc[result["metrics"].version.eq("Equal weight"), "cagr"].iloc[0]
            self.assertNotAlmostEqual(ew(first), ew(second))
            loaded = load_final_books(save_final_books(first, root / "artifacts"))
            self.assertNotIn("predictions", loaded)
            self.assertEqual(len(list_final_books(root / "artifacts")), 1)

    def _render(self, root):
        script = (
            "from app import render_final_test\n"
            "from research.store import ExperimentStore\n"
            f"render_final_test(ExperimentStore(root={str(root)!r}, project_root={str(root)!r}))\n"
        )
        with patch.dict(os.environ, {"MLFACTOR_READ_ONLY": "1"}), patch(
                "research.models.predict_model", side_effect=AssertionError("browsing fitted")), patch(
                "research.final_books.score_final_books", side_effect=AssertionError("browsing fitted")):
            return AppTest.from_string(script).run(timeout=30)

    def test_empty_final_tab_has_no_table_and_keeps_the_pending_sentence(self):
        with tempfile.TemporaryDirectory() as folder:
            app = self._render(Path(folder))
            self.assertFalse(app.exception)
            self.assertFalse(app.dataframe)
            self.assertTrue(any(
                "Final frozen-model test evaluation is pending" in item.value for item in app.info))
            self.assertTrue(any("no 2010–2018 row" in item.value for item in app.caption))

    def test_saved_study_is_visible_before_the_archive_control(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _fake_study(root)
            app = self._render(root)
            self.assertFalse(app.exception)
            self.assertTrue(app.dataframe)
            self.assertTrue(any("Book" in frame.value for frame in app.dataframe))
            self.assertFalse(any("Strategy" in frame.value for frame in app.dataframe))
            self.assertTrue(any(
                "Final frozen-model test evaluation is pending" in item.value for item in app.info))
            self.assertTrue(app.checkbox(key='final_main_families_only').value)
            app.checkbox(key='final_main_families_only').uncheck().run()
            app.checkbox(key="show_test_history").check().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Strategy" in frame.value for frame in app.dataframe))
            self.assertTrue(any(
                "Final frozen-model test evaluation is pending" in item.value for item in app.info))


if __name__ == "__main__":
    unittest.main()
