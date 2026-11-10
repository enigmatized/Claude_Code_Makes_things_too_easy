"""Adapters import the existing, independently tested accounting and rules.

The earlier notebooks and their modules remain unchanged. Only this bridge
knows their historical folder names; the runner and UI use normal imports.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for directory in ["part0_data_and_benchmark", "0_1_Basic_Strategies"]:
    path = str(PROJECT_ROOT / directory)
    if path not in sys.path:
        sys.path.insert(0, path)

from pipeline import FEATURES, KEYS, TARGET, load_panel, performance, portfolio_returns, return_month
from strategies import LABELS, build_signals, strategy_weights
from hybrid import HYBRID_LABELS, hybrid_weights, rank_training_volatility
from downside import SIZING_LABELS, build_risk_signals, downside_statistics, risk_adjust_weights
