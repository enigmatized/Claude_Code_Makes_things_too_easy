"""Read-only evaluation labels and historical non-ML test archives."""
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .legacy import PROJECT_ROOT
from .store import digest_file


def evaluation_scope(config, manifest):
    config = asdict(config) if is_dataclass(config) else config
    evaluation = config["evaluation"]
    if evaluation not in {"training_validation", "test"}:
        raise ValueError("Unknown evaluation period.")
    first = pd.Timestamp(manifest["first_return_month"])
    last = pd.Timestamp(manifest["last_return_month"])
    title = "TRAINING VALIDATION" if evaluation == "training_validation" else "TEST RESULTS"
    period = f"{first:%b %Y} – {last:%b %Y}"
    if config.get("include_ml", False):
        if config.get("fit_policy", "expanding") == "frozen_at_origin":
            protocol = "ML models fit once before this evaluation starts; evaluation outcomes do not enter later model fitting."
            fitting = "Frozen models"
        else:
            protocol = (f"ML models refit every {config.get('refit_months', 12)} months on then-known outcomes. "
                        "Earlier evaluation outcomes can enter later fits.")
            fitting = "Expanding research fits"
        if evaluation == "training_validation":
            protocol += " These are chronological validation predictions within the training dataset, not predictions evaluated on their own fit rows."
        elif fitting == "Expanding research fits":
            protocol += " This is exploratory test evaluation; it is different from the final frozen-model protocol."
    else:
        fitting = "Non-ML rules"
        protocol = "Non-ML trading rules use only historical information available at each rebalance."
    purpose = ("Used to compare research choices. Test-period returns are not included in this section."
               if evaluation == "training_validation" else
               "The test period was viewed earlier. A test-period label alone does not establish a completed final submission experiment.")
    return {"evaluation": evaluation, "title": title, "period": period,
            "fitting": fitting, "protocol": protocol, "purpose": purpose}


ARCHIVE_FILES = (
    "0_1_Basic_Strategies/outputs/downside/strategy_comparison.csv",
    "0_1_Basic_Strategies/outputs/monthly_strategy_returns.csv",
    "0_1_Basic_Strategies/outputs/hybrid/hybrid_monthly_returns.csv",
    "0_1_Basic_Strategies/outputs/downside/new_strategy_monthly_returns.csv",
    "0_1_Basic_Strategies/outputs/downside/run_manifest.json",
    "0_2_Strategy_Runner/outputs/ratio_pools_test_comparison/strategy_comparison.csv",
    "0_2_Strategy_Runner/outputs/ratio_pools_test_comparison/monthly_returns.csv",
    "0_2_Strategy_Runner/outputs/ratio_pools_test_comparison/manifest.json",
)


def load_archived_test_results(root=PROJECT_ROOT):
    """Read thirteen recorded test portfolios; never replay rules or fit models.

    Combine earlier basic/hybrid/downside and ratio-screen reports only after
    reconciling their shared strategies. Original files remain untouched.
    """
    root = Path(root)
    missing = [name for name in ARCHIVE_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Historical test archive is incomplete: {missing[0]}")
    read = lambda name: pd.read_csv(root / name, float_precision="round_trip")
    basic = read(ARCHIVE_FILES[0])
    basic = basic.loc[basic.split.eq("test")].copy()
    monthly = pd.concat([read(name) for name in ARCHIVE_FILES[1:4]], ignore_index=True)
    monthly = monthly.loc[monthly.split.eq("test")].copy()
    ratio = read(ARCHIVE_FILES[5])
    ratio_monthly = read(ARCHIVE_FILES[6])
    manifest = json.loads((root / ARCHIVE_FILES[7]).read_text())
    if manifest["config"]["evaluation"] != "test" or manifest["config"]["include_ml"]:
        raise ValueError("Expected a non-ML test archive.")
    basic_manifest = json.loads((root / ARCHIVE_FILES[4]).read_text())
    for name, digest in manifest["source_sha256"].items():
        if basic_manifest["sources"].get(name) != digest:
            raise ValueError("Historical archives used different input data.")
    for frame in [monthly, ratio_monthly]:
        frame["return_month"] = pd.to_datetime(frame.return_month)
        frame["snapshot_date"] = pd.to_datetime(frame.snapshot_date)
    shared = set(basic.strategy) & set(ratio.strategy)
    for strategy in shared:
        columns = ["cagr", "sharpe", "sortino", "max_drawdown", "avg_one_way_turnover"]
        np.testing.assert_allclose(basic.set_index("strategy").loc[strategy, columns].astype(float),
                                   ratio.set_index("strategy").loc[strategy, columns].astype(float),
                                   rtol=1e-12, atol=1e-12)
        left = monthly.loc[monthly.strategy.eq(strategy)].sort_values("return_month")
        right = ratio_monthly.loc[ratio_monthly.strategy.eq(strategy)].sort_values("return_month")
        if not left.return_month.reset_index(drop=True).equals(right.return_month.reset_index(drop=True)):
            raise ValueError("Archived test calendars differ.")
        np.testing.assert_allclose(left[["simple_return", "wealth", "drawdown"]],
                                   right[["simple_return", "wealth", "drawdown"]], rtol=1e-12, atol=1e-12)
    metrics = pd.concat([basic.assign(source_archive="basic/hybrid/downside"),
                         ratio.assign(source_archive="ratio pools")], ignore_index=True).drop_duplicates("strategy")
    monthly = pd.concat([monthly, ratio_monthly], ignore_index=True).drop_duplicates(["strategy", "return_month"])
    expected = pd.date_range("2010-01-31", "2018-12-31", freq="ME")
    for _, rows in monthly.groupby("strategy"):
        if not pd.DatetimeIndex(rows.return_month.sort_values()).equals(expected):
            raise ValueError("Historical test archive must contain the full 2010–2018 window.")
    if set(metrics.strategy) != set(monthly.strategy):
        raise ValueError("Historical metrics and return histories must cover the same strategies.")
    metrics["split"] = "test"
    monthly["split"] = "test"
    metrics = metrics.drop(columns=["label", "variant"], errors="ignore")
    monthly = monthly.drop(columns=["label", "variant"], errors="ignore")
    return {"metrics": metrics.reset_index(drop=True),
            "monthly": monthly.sort_values(["strategy", "return_month"]).reset_index(drop=True),
            "manifest": manifest,
            "file_sha256": {name: digest_file(root / name) for name in ARCHIVE_FILES}}
