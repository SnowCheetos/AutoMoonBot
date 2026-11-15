# Options, futures and margin

The backtester trades listed options and futures alongside stock, with
expiry, exercise and assignment, daily futures settlement, and margin
requirements that can force liquidations. The simulator is
`moonrs::backtest` (Rust), exposed to Python as `automoonbot._moonrs.Simulator`.
Data comes from `automoonbot.data.providers.massive`.

Everything below keeps the backtester's timing rule: a decision made after
bar `t` closes, from data known by then, executes at the open of bar `t + 1`.

## Instruments

Each column of a market is one instrument (`MarketPanel.with_instruments`,
or `with_contracts` to read option terms from the store):

| kind | units | value in equity | notes |
|---|---|---|---|
| `equity` | shares | shares × price | dividends, splits, borrow on shorts |
| `index` | — | — | a reference level (e.g. SPX) for index options; not tradable |
| `option` | contracts | contracts × multiplier × price | `underlying` column, `right`, `strike`, `expiry` (a bar index), `style`, `settlement`, `deliverable` |
| `future` | contracts | 0 (gains settle to cash daily) | `multiplier`, `expiry`, `settlement`, initial and maintenance margin as fractions of notional |

Strategies give targets as weights of equity (`step`) or as unit counts
(`step(..., units=True)`, `run(..., units=True)`), which is what option
strategies want ("short one call"). For a future, a weight is its notional
over equity. Contracts are rounded to whole lots.

## Lifecycle

* **Premium.** Options are bought and sold for their premium. Futures cost
  nothing to open; they settle to the open before trading and to the close
  after it, every bar.
* **Expiry.** After the close of the last trading bar, an option at least
  `exercise_threshold` (default $0.01 per delivered unit) in the money is
  exercised if long and assigned if short. Physical settlement delivers
  `deliverable` units of the underlying at the strike. Cash settlement pays
  the intrinsic value at the close or at the next open (AM-settled index
  options). Futures settle at their own close or at the underlying's next
  open.
* **Early exercise** (American options, on by default) follows the rational
  holder, for long and short positions alike. A call is exercised before an
  ex-dividend bar when the dividend is larger than its time value. A put is
  exercised when its time value falls to the threshold.
* **Corporate actions.** A split of the underlying adjusts option terms the
  way the OCC does: the strike and deliverable change so the contract's
  value is unchanged.
* **Delisting.** A future that stops being listed before expiry is settled
  at its last close with the usual haircut. An option whose trades stop
  stays listed to expiry at its last trade price, so a quiet contract is
  held, not force-closed.

## Costs

Options pay a per-contract commission (default $0.65) and a half-spread of
the larger of 100 bps of the premium and $0.005. Futures pay a per-contract
commission (default $2.25) and the stock half-spread in bps. When quotes are
available, `MarketPanel.with_half_spread` supplies an absolute half-spread
per bar (known before the open), which replaces the bps model.

## Margin

`Simulator(..., margin=...)` chooses the model. All requirements scale
linearly when every position is scaled, which the simulator relies on to
size cutbacks.

* `"none"` (default): no requirement. Leverage is limited only by
  `max_gross`. Stock-only backtests behave exactly as before.
* `"reg_t"`: Regulation T / FINRA 4210, position by position. Stock: 50%
  initial; 25% maintenance long, 30% short. Long options: full premium (75%
  beyond nine months). Short options: premium plus 20% of the underlying
  (15% for broad indices) less the out-of-the-money amount, and at least
  premium plus 10% of the underlying (calls) or of the strike (puts).
  Futures: their own fractions of notional.
* `"portfolio"`: risk-based, in the style of OCC TIMS / FINRA portfolio
  margin. Positions are grouped by root underlying and revalued over ±15%
  moves (−8%/+6% for broad indices) on 11 points, optionally with volatility
  shifts. Options are repriced with Black–Scholes (Black-76 on futures) at
  the volatility implied by their own price, never below intrinsic for
  American options. A group needs its worst loss, at least $37.50 per
  contract. Hedges offset within a group, never across groups.

Margin works in two places:

1. **Initial.** Before fills, the trade is scaled back by bisection until
   the book meets the initial requirement. A trade that does not raise an
   existing deficiency is always allowed.
2. **Maintenance.** At every close, if equity is below the maintenance
   requirement, the next open cuts every tradable position by the same
   fraction until it is met. The strategy's target for that bar is not
   executed. The report shows `margin_call` and `liquidated`.

## Option data

`MassiveProvider` (Massive, formerly Polygon.io) lists contracts and
downloads unadjusted daily bars. The API key is read from `MASSIVE_API_KEY`
and sent only in the `Authorization` header: it is never put in URLs, cache
keys, stored rows, logs or reprs.

`ingest_chain` keeps the universe point-in-time. Downloading only strikes
that were ever near the money would leak the future: a listed 120 call
while the stock is at 100 says it will rally. Instead, a contract is
admitted on the first session (at most `max_days` before expiration) whose
underlying close is within `moneyness` of its strike. Its bars and terms
are stored from that session on, and both are known only after that close.

Option bars end at 16:15 New York and stock bars at 16:00, so build the
panel with `MarketPanel.from_store(..., align="session")`. That keys rows by
session date, and each row's decision time is the latest knowledge time in
it. Then call `with_contracts(store)`, which maps each expiration date to
its bar index.

On the free plan: two years of history, 5 requests a minute, and trades
only. Two years of SPY monthly chains with $5 strikes within 10% is about
1,500 requests, roughly 5 hours, and an interrupted run resumes from the
HTTP cache.

## Limitations

* **Stale marks.** Free-tier bars are trade aggregates. An illiquid
  contract's close can lag the underlying's close by hours. Marks, implied
  vols and margin are only as good as those prices. Quotes (bid/ask) need
  a paid plan, or a daily snapshot recorded going forward.
* **Adjusted contracts.** When a split or special dividend makes the OCC
  re-list a contract under a new symbol, the provider sees a new contract
  and the old one stops trading. The simulator values the old one at its
  last trade until expiry. SPY and index options are unaffected.
* **Reg T pairs nothing.** Spreads and covered writes are charged as if
  each leg stood alone. Use portfolio margin for hedged books.
* **Exercise is rational and immediate.** No exercise-by-exception
  overrides, no random early assignment, and no pin risk around the strike.
* **No index options yet.** Massive's index-options feed (SPX, cash
  settled) is a separate product. The instruments for it (`index`, cash
  settlement at the next open) are already supported.
