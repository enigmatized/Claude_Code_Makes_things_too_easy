"""Plain-language portfolio rules shared by the app and run manifests."""
from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyExplanation:
    purpose: str
    universe: str
    selection: str
    weighting: str
    history: str
    fallback: str
    example: str


def explanation_for(key):
    if key == "annual_consistency_momentum_cash":
        return StrategyExplanation(
            "Hold historically consistent annual winners only while their momentum is positive; allow cash.",
            "Freeze the pool at evaluation start using only then-known training returns. Require at least three complete calendar years and more than half positive; flat years count as nonpositive.",
            "Each month keep only pool stocks with the configured number of consecutive known returns (default twelve) and strictly positive compounded momentum over that window. Zero, negative and unavailable signals get zero stock weight.",
            "If P of N currently available pool stocks qualify, invest P/N of total capital, weighted in proportion to their positive momentum. Keep the rest in cash at an assumed 0% monthly return.",
            "Calendar-year returns compound all twelve January--December earned returns. Incomplete years are excluded. The annual pool stays fixed; lagged momentum updates monthly using its separate configurable lookback.",
            "No positive signals, no available pool stocks, or an empty historical pool means 100% cash. There is no fallback purchase of negative-momentum stocks.",
            "With four pool stocks and two positive momentum scores of 10% and 30%, invest 12.5% and 37.5%; hold 50% cash. The ML version multiplies stock weights by (1 - strength × P(loss)) and adds the reduction to cash.")
    history = "Twelve consecutive known monthly returns. Momentum compounds those returns; volatility uses their sample standard deviation."
    fallback = "With no positive momentum, equal-weight eligible stocks. With no eligible history, equal-weight available stocks."
    if key == "equal_weight_universe":
        return StrategyExplanation("The reference portfolio for the available stock universe.",
            "All stocks with a row at this month's snapshot.", "Keep every available stock.",
            "Divide capital equally; rebalance monthly.", "No return-history requirement.",
            "No screening fallback is needed.", "With 100 stocks, each receives 1% of capital.")
    if key == "equal_weight_eligible":
        return StrategyExplanation("Isolate the effect of requiring historical data.",
            "All available stocks.", "Keep every stock with 12 consecutive known monthly returns.",
            "Equal weights.", history, "If none qualifies, equal-weight available stocks.",
            "If 80 of 100 stocks have enough history, each of those 80 receives 1.25%.")
    if key in {"high_volatility", "low_volatility"}:
        high = key == "high_volatility"
        return StrategyExplanation("Compare a portfolio of stocks with larger recent fluctuations." if high else "Compare a portfolio of stocks with smaller recent fluctuations.",
            "Available stocks with enough history.",
            f"Each month choose the {'highest' if high else 'lowest'} 20% by trailing 12-month volatility. Round up; ties use lower stock ID.",
            "Equal weights among the selected stocks.", history,
            "If none has enough history, equal-weight available stocks.",
            "From 100 eligible stocks, select 20 and assign 5% to each.")
    if key in {"momentum_equal_weight", "momentum_weighted"}:
        equal = key == "momentum_equal_weight"
        return StrategyExplanation("Buy recent winners, then compare how capital is distributed.",
            "Available stocks with enough history.",
            "Choose the highest positive 12-month momentum, up to 20% of eligible stocks. Hold fewer if fewer signals are positive.",
            "Equal weights in the chosen basket." if equal else "Weights are proportional to positive momentum, then normalized to 100%.",
            history, fallback,
            "Two chosen stocks receive 50% each." if equal else "Momentum scores 10% and 30% produce weights 25% and 75%.")
    metric = "Sharpe" if "sharpe" in key else "Sortino" if "sortino" in key else "volatility"
    pool = (f"Freeze the top 20% by historical {metric} known at the first evaluation snapshot. "
            "Use only then-realized training returns and at least 12 observed months per stock. Membership stays fixed.")
    if metric in {"Sharpe", "Sortino"}:
        pool += " Ratios use a zero reference return; undefined ratios are excluded and recorded."
    selection = "Each month choose the highest positive 12-month momentum, up to 20% of eligible pool members."
    weighting = "Allocate in proportion to positive momentum, then normalize to 100%."
    example = "A pool near 200 eligible stocks produces a momentum basket near 40; momentum scores determine their weights."
    if key == "training_high_vol_equal_weight":
        selection, weighting = "Keep all currently eligible pool members.", "Equal weights across eligible pool members."
        example = "With 200 eligible pool members, each receives 0.5%."
    elif key == "training_high_vol_momentum_equal":
        weighting = "Equal weights inside the selected positive-momentum basket."
        example = "With 40 selected stocks, each receives 2.5%."
    if key.startswith("hybrid_"):
        downside = key == "hybrid_downside_3m_sizing"
        weighting = ("Positive momentum divided by " + ("custom downside-weighted risk" if downside else "ordinary volatility")
            + " estimated from the last 3 monthly returns; normalize to 100%. Use a 1% monthly risk floor in the same units.")
        history += " Sizing risk uses 3 monthly observations."
        if downside:
            history += " Custom risk adds mean squared negative returns to sample variance before taking the square root; all three months count."
        example = "For equal momentum, the stock with lower sizing risk receives more capital. No position cap is imposed."
    pool_fallback = "If no member has enough history, equal-weight available pool members. No available pool member is an explicit error."
    if key != "training_high_vol_equal_weight":
        pool_fallback = "If no momentum is positive, equal-weight eligible pool members. " + pool_fallback
    if key.startswith("hybrid_"):
        pool_fallback += " Risk sizing preserves the parent fallback."
    return StrategyExplanation(f"Use an initial {metric} screen before monthly allocation.", pool,
        selection, weighting, history, pool_fallback, example)
