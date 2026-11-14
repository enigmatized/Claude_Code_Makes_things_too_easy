"""Causal signals, fixed membership, cash accounting and source-free readers."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from acceleration_publication import export_acceleration
from published_results import PublishedStrategies
from research.basket_momentum import FEATURE, concentrated_basket_weights, _digest
from research.legacy import KEYS, TARGET
from research.momentum_acceleration import (
    METHODS,
    VARIANTS,
    _check,
    acceleration_signals,
    adjust_for_deceleration,
    create_acceleration_study,
    load_acceleration_study,
)


class AccelerationRuleTests(unittest.TestCase):
    def history(self):
        dates = pd.date_range("2004-11-30", periods=4, freq="ME")
        frame = pd.MultiIndex.from_product([dates, range(3)], names=KEYS).to_frame(
            index=False
        )
        frame[TARGET] = [
            0.10,
            0.10,
            -0.04,
            0.04,
            0.08,
            0.02,
            0.9,
            0.8,
            0.7,
            0.6,
            0.5,
            0.4,
        ]
        selected = pd.DataFrame(
            {"date": [dates[2]] * 3, "stock_id": [0, 1, 2], FEATURE: [7, 7, 9]}
        )
        return frame, selected

    def test_earned_months_and_current_membership_ignore_forward_outcomes(self):
        history, selected = self.history()
        signal = acceleration_signals(history, selected)
        np.testing.assert_allclose(signal.stock_return_now, [0.04, 0.08, 0.02])
        np.testing.assert_allclose(signal.stock_return_previous, [0.10, 0.10, -0.04])
        np.testing.assert_allclose(signal.stock_acceleration, [-0.06, -0.02, 0.06])
        np.testing.assert_allclose(signal.basket_return_now, [0.06, 0.06, 0.02])
        np.testing.assert_allclose(signal.basket_return_previous, [0.10, 0.10, -0.04])
        np.testing.assert_allclose(signal.basket_acceleration, [-0.04, -0.04, 0.06])
        history.loc[history.date.ge(selected.date.iloc[0]), TARGET] = 99
        pd.testing.assert_frame_equal(signal, acceleration_signals(history, selected))
        older = selected.assign(date=pd.Timestamp("2004-12-31"), **{FEATURE: [0, 1, 0]})
        joined = acceleration_signals(
            history, pd.concat([older, selected], ignore_index=True)
        )
        pd.testing.assert_frame_equal(
            signal,
            joined.loc[joined.date.eq(selected.date.iloc[0])].reset_index(drop=True),
        )

    def test_original_book_uses_equal_mean_signal_and_unequal_actual_positions(self):
        history, selected = self.history()
        # Four slots; two positive names in one positive group, two in a negative group.
        date = selected.date.iloc[0]
        history = pd.concat(
            [history, history.loc[history.stock_id.eq(2)].assign(stock_id=3)],
            ignore_index=True,
        )
        history.loc[
            history.date.eq(pd.Timestamp("2004-12-31")) & history.stock_id.ge(2), TARGET
        ] = -0.02
        history.loc[
            history.date.eq(pd.Timestamp("2004-12-31")) & history.stock_id.eq(0), TARGET
        ] = 0.10
        history.loc[
            history.date.eq(pd.Timestamp("2004-12-31")) & history.stock_id.eq(1), TARGET
        ] = 0.30
        selected = pd.DataFrame(
            {"date": [date] * 4, "stock_id": range(4), FEATURE: [0, 0, 1, 1]}
        )
        panel = history.loc[history.date.eq(date), KEYS + [TARGET]]
        weights, signals, _ = concentrated_basket_weights(history, panel, selected, 1)
        np.testing.assert_allclose(
            signals.loc[signals[FEATURE].eq(0), "basket_momentum"], 0.20
        )
        np.testing.assert_allclose(weights.weight, [0.125, 0.375, 0, 0])

    def test_haircuts_compound_without_reallocation_shorts_or_resurrected_positions(
        self,
    ):
        date = pd.Timestamp("2005-01-31")
        weights = pd.DataFrame(
            {
                "date": [date] * 5,
                "stock_id": range(5),
                "weight": [0.2, 0.3, 0, 0.1, 0.2],
            }
        )
        signals = weights[KEYS].assign(
            stock_acceleration=[-0.1, 0.1, -0.1, np.nan, -0.1],
            basket_acceleration=[0.1, -0.1, -0.1, np.nan, -0.1],
        )
        expected = (
            [0.2, 0.3, 0, 0.1, 0.2],
            [0.1, 0.3, 0, 0.1, 0.1],
            [0.2, 0.15, 0, 0.1, 0.1],
            [0.1, 0.15, 0, 0.1, 0.05],
        )
        for variant, values in zip(VARIANTS, expected):
            adjusted = adjust_for_deceleration(weights, signals, variant)
            np.testing.assert_allclose(adjusted.weight, values)
            self.assertTrue(adjusted.weight.between(0, weights.weight).all())
            pd.testing.assert_frame_equal(
                weights, adjust_for_deceleration(weights, signals, variant, strength=0)
            )
        for bad in (
            signals.assign(stock_id=99),
            pd.concat([signals, signals.iloc[:1]]),
        ):
            with self.assertRaises(ValueError):
                adjust_for_deceleration(weights, bad, VARIANTS[-1])
        for strength in (-0.1, 1.1, np.nan):
            with self.assertRaises(ValueError):
                adjust_for_deceleration(weights, signals, VARIANTS[-1], strength)

    def test_missing_observation_stays_missing_and_empty_screen_keeps_cash(self):
        history, selected = self.history()
        history = history.loc[
            ~(history.stock_id.eq(0) & history.date.eq(pd.Timestamp("2004-11-30")))
        ]
        signals = acceleration_signals(history, selected)
        self.assertTrue(np.isnan(signals.stock_acceleration.iloc[0]))
        self.assertTrue(
            signals.loc[signals[FEATURE].eq(7), "basket_acceleration"].isna().all()
        )
        empty = acceleration_signals(history, selected.iloc[:0])
        weights = selected[KEYS].assign(weight=0.0)
        pd.testing.assert_frame_equal(
            weights, adjust_for_deceleration(weights, empty, VARIANTS[-1])
        )


class AccelerationPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        pd.DataFrame({"unchanged": [1]}).to_parquet(
            self.source / "old.parquet", index=False
        )
        meta = {
            "publication_schema": 1,
            "tabs": {"strategies": {}},
            "file_sha256": {"old.parquet": _digest(self.source / "old.parquet")},
        }
        (self.source / "manifest.json").write_text(json.dumps(meta))
        self.study = self.root / "study"
        self.study.mkdir()
        (self.study / "study.json").write_text(
            json.dumps({"strength": 0.5, "new_directional_fits": 0})
        )
        for window, date in [("training", "2009-12-31"), ("test", "2010-01-31")]:
            date = pd.Timestamp(date)
            stage = self.study / window
            stage.mkdir()
            monthly, metrics, signals = [], [], []
            for method in METHODS:
                for variant in VARIANTS:
                    monthly.append(
                        dict(
                            snapshot_date=date - pd.offsets.MonthEnd(1),
                            return_month=date,
                            simple_return=0.01,
                            wealth=1.01,
                            drawdown=0.0,
                            cash_weight=0.5,
                            method=method,
                            variant=variant,
                        )
                    )
                    metrics.append(
                        dict(
                            method=method,
                            variant=variant,
                            cagr=0.12,
                            annualized_log_growth=0.113,
                            annualized_volatility=0.2,
                            sharpe=0.6,
                            max_drawdown=0.0,
                            avg_cash_weight=0.5,
                            avg_one_way_turnover=0.1,
                        )
                    )
                signals.append(
                    dict(
                        date=date - pd.offsets.MonthEnd(1),
                        stock_id=1,
                        method=method,
                        correlation_basket=0,
                        stock_return_now=0.1,
                        stock_return_previous=0.2,
                        stock_acceleration=-0.1,
                        basket_return_now=0.1,
                        basket_return_previous=0.2,
                        basket_acceleration=-0.1,
                        original_weight=0.5,
                        **dict(zip(VARIANTS[1:], [0.25, 0.25, 0.125])),
                    )
                )
            scope = dict(
                window=window,
                test_csv_loaded=window == "test",
                first_return_month=str(date.date()),
                last_return_month=str(date.date()),
                months=1,
            )
            record = {"scope": scope, "frames": {}}
            for name, data in [
                ("metrics", metrics),
                ("monthly", monthly),
                ("signals", signals),
            ]:
                path = stage / (name + ".parquet")
                pd.DataFrame(data).to_parquet(path, index=False)
                record["frames"][name] = _digest(path)
            (stage / "complete.json").write_text(json.dumps(record))

    def tearDown(self):
        self.temp.cleanup()

    def test_source_free_published_and_local_views_match_and_show_both_methods(self):
        output = export_acceleration(self.source, self.study, self.root / "published")
        publication = PublishedStrategies(output)
        reader = publication.analysis_section("acceleration")
        self.assertEqual(
            _digest(output / "old.parquet"), _digest(self.source / "old.parquet")
        )
        script = (
            "from acceleration_ui import render_acceleration\n"
            "from published_results import PublishedStrategies\n"
            f'p=PublishedStrategies({str(output)!r}).analysis_section("acceleration")\n'
            'render_acceleration(published=p)\nrender_acceleration(published=p,window="test")'
        )
        with patch("pandas.read_csv", side_effect=AssertionError("No datasets")), patch(
            "research.store.ExperimentStore.__init__",
            side_effect=AssertionError("No catalog"),
        ), patch(
            "sklearn.cluster.KMeans.fit_predict",
            side_effect=AssertionError("No clustering"),
        ), patch(
            "research.momentum_acceleration.create_acceleration_study",
            side_effect=AssertionError("No replay"),
        ):
            for window in ("training", "test"):
                saved = reader.study(window)
                local = load_acceleration_study(self.study, window)
                for name in ("monthly", "metrics", "signals"):
                    pd.testing.assert_frame_equal(saved[name], local[name])
            app = AppTest.from_string(script).run(timeout=20)
            self.assertFalse(app.exception)
            self.assertTrue(
                any("Training data only" in item.value for item in app.info)
            )
            self.assertTrue(any("Official test" in item.value for item in app.info))
            app.selectbox(key="acceleration_training_method").select("Ward").run()
            self.assertFalse(app.exception)
            app.selectbox(key="acceleration_test_return").select(
                "Cumulative log return"
            ).run()
            self.assertFalse(app.exception)
            app.selectbox(key="acceleration_test_return").select(
                "Monthly log return"
            ).run()
            self.assertFalse(app.exception)
        local_script = (
            "from types import SimpleNamespace\nfrom pathlib import Path\n"
            "from acceleration_ui import render_acceleration\n"
            f"render_acceleration(SimpleNamespace(root=Path({str(self.root)!r})))"
        )
        # Match the research store's artifact layout without loading its catalog.
        directory = self.root / "acceleration_studies"
        directory.mkdir()
        self.study.rename(directory / "study")
        app = AppTest.from_string(local_script).run(timeout=20)
        self.assertFalse(app.exception)

    def test_timing_scope_integrity_and_deploy_computation_guards(self):
        study = load_acceleration_study(self.study)
        for mutation in ("test", "timing", "declared_dates", "cross_boundary"):
            bad = {
                **study,
                "scope": dict(study["scope"]),
                "monthly": study["monthly"].copy(),
            }
            if mutation == "test":
                bad["scope"]["test_csv_loaded"] = True
            if mutation == "timing":
                bad["monthly"]["snapshot_date"] = bad["monthly"].return_month
            if mutation == "declared_dates":
                bad["scope"]["first_return_month"] = "2008-12-31"
            if mutation == "cross_boundary":
                bad["monthly"]["return_month"] = pd.Timestamp("2010-01-31")
            with self.assertRaises(ValueError):
                _check(bad, "training")
        with patch.dict("os.environ", {"MLFACTOR_READ_ONLY": "1"}):
            with self.assertRaises(RuntimeError):
                create_acceleration_study(self.root)
        path = self.study / "training/monthly.parquet"
        path.write_bytes(path.read_bytes() + b"damaged")
        with self.assertRaisesRegex(ValueError, "Damaged"):
            load_acceleration_study(self.study)


if __name__ == "__main__":
    unittest.main()
