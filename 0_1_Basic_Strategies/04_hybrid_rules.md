# Training high volatility, followed by monthly momentum

This addition interprets the requested high-volatility screen as a stock list
learned from training and then frozen for the testing period. It differs from
the original `high_volatility` strategy, which rebuilds its volatility basket
each month using a rolling twelve-month history.

## Fixed specification

1. Use the supplied training labels earned January 2000–December 2009. For each
   stock with at least twelve observed training returns, compute sample monthly
   return volatility, annualized by sqrt(12). No future-survival requirement or
   missing-return filling is used.
2. Rank these stocks by full-training volatility and freeze the highest 20%,
   rounding the name count upward. Ties go to the lower stock ID. The list is
   known at the December 2009 rebalance and is not updated using test returns.
3. Each test month, consider only currently available frozen-list stocks with
   twelve consecutive already-realized monthly returns. Calculate their trailing
   compounded twelve-month momentum, including the last realized month.
4. Select the strongest positive-momentum names, targeting 20% of this eligible
   frozen list and rounding upward. If fewer names have positive momentum, hold
   those names. Normalize their positive momentum scores into capital weights.
   There is no weight cap, borrowing, or shorting.
5. If none has positive momentum, equal-weight eligible frozen-list stocks. If
   none has enough history, equal-weight currently available frozen-list stocks.
   Never allocate outside the frozen list. If no frozen-list stock is present,
   stop explicitly; switching to cash would need a separate rule.

The two-stage 20% screens make a typical hybrid basket about 4% of the initial
stock universe. It can therefore be much more concentrated than the original
momentum basket. The percentage is retained from the initial experiment, not
searched after seeing results.

## Controls and timing

Add two controls: equal weights across eligible frozen-list stocks, and equal
weights on the same positive-momentum basket used by score weighting. These
distinguish the static training screen, monthly selection, and capital weighting.
Compare them against all original strategies on January 2010–December 2018.

Do not apply the final 2000–2009 stock list to earlier training returns and call
that an out-of-sample training backtest. Membership would depend on later years.
An honest earlier process check would need to refit the volatility screen at
successive historical origins. This addition reports training classification
statistics and later test performance, without inventing that earlier test.

For the January 2010 allocation, monthly momentum uses January–December 2009
returns. Later allocations can use earlier test returns after those returns
have been realized; the frozen stock list never changes. Raw extreme returns
remain in both the volatility estimate and the earned portfolio returns.

The test period has already been examined in this project. The hybrid is a
new, explicitly requested historical experiment, not a fresh untouched test.
Original strategy results are preserved, and outputs for this addition go into
`outputs/hybrid/`.
