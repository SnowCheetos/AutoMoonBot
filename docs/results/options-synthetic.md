# Option writing on SPY, 1996–2026, with synthetic Black-Scholes options

`experiments/options_synthetic.py`, run on Yahoo data through 2026-11-13.
The run covers 7,585 sessions, from 1996-09-23 to 2026-11-13.

## Bottom line

* **The machinery checks out.**
  * With early exercise turned off, an at-the-money covered call and an at-the-money put write earn the same to within 0.01% a year (10.64% vs 10.65%, frictionless, in a run at ATM vol = 0.9 × VIX). Put-call parity survives the whole pipeline: pricing, rolls, exercise, dividends and margin.
  * The synthetic put write tracks the real Cboe S&P 500 PutWrite index (PUT) month by month, with correlation 0.95 and similar volatility (10–12% vs 9–12%).
* **The level is uncertain by about 2% a year.**
  * One parameter maps VIX to at-the-money implied volatility: ATM vol = 0.86 × VIX. It was fitted so that the put write matches PUT's average return over 1996–2010.
  * On 2011–2026, which was not used in the fit, the synthetic put write earns 9.6% a year against PUT's 7.8%.
  * The mapping drifted: the Cboe SKEW index rose from about 113 to about 147, and VIX moved further above at-the-money volatility. The 1996–2010 data cannot pin down that dependence. Fitting it to the later years would be peeking.
  * The real Massive chains for 2024–2026 can measure it honestly.
* **Strategy result.**
  * 1996–2010: option writing beat SPY on return and on Sharpe over T-bills (0.35–0.60 vs 0.24). SPY went through two 50% crashes in that span.
  * 2011–2026: every writer earned less than SPY (8.5–10.9% vs 13.9% a year) with similar or lower excess Sharpe (0.62–0.75 vs 0.76). These writer numbers are optimistic by roughly the 1.7% a year above.
  * Writing options lowered volatility and drawdowns (max drawdown 28–41% vs 55%). It did not add risk-adjusted return in the second half.
* **Out-of-the-money results are not trustworthy yet.**
  * Only the at-the-money level is calibrated. The skew shape is a typical-looking guess.
  * The 2% out-of-the-money put write earns 10.1% a year with the skew and 6.1% without it (1996–2010, frictionless). The assumption is larger than the effect.

## Setup

**Underlying and cash**
* SPY daily bars, as traded, with Yahoo's splits and dividends.
* Cash earns the 13-week T-bill rate each day. This uses the new rate series in the simulator; Yahoo `^IRX`, converted from discount yield.

**Options**
* Synthetic monthly SPY options:
  * listed at the close of the previous monthly expiration, with strikes from that close rounded to $1;
  * American, physically settled, 100 shares;
  * priced with Black-Scholes (carry = T-bill rate minus trailing dividend yield), floored at intrinsic and at $0.01.
* Opens are priced with the VIX open and the previous close's rate. Every price uses only data from its own time or earlier (`backtest.synthetic`, tested).

**Volatility, three ways**
* `vix_skew`: ATM = 0.86 × VIX, plus an S&P-like skew in standardized moneyness (puts 25 delta out of the money about 30% richer, calls about 13% cheaper).
* `vix_flat`: the same ATM level, no skew.
* `realized_flat`: SPY's trailing 21-session volatility, no skew. This is the naive Black-Scholes model.

**Strategies** (`OptionWrite`)
* Each writes the next monthly contract on 100% of equity at each expiration and holds it to expiration:
  * covered call: long SPY, short calls;
  * cash-secured put write.
* Each comes at the money and 2% out of the money.
* Buy-and-hold SPY tops up to fully invested on the same roll dates.

**Simulation**
* $1M, portfolio margin, one out-of-sample path with 20 causality probes per strategy.

**Costs**
* frictionless;
* tight: SPY 0.5 bp half-spread and 0.2 bp commission; options $0.65 a contract and a half-spread of max(1%, $0.005);
* wide: option half-spread max(3%, $0.05).

## Calibration against the Cboe PutWrite index

At-the-money put write, frictionless, monthly log returns, annualized.

| model | period | synthetic | PUT index | vol (synthetic / PUT) | tracking error | correlation |
|---|---|---:|---:|---:|---:|---:|
| vix_skew | 1996–2010 (fit) | 8.4% | 8.4% | 11.5% / 12.2% | 3.5% | 0.96 |
| vix_skew | 2011–2026 | 9.6% | 7.8% | 10.0% / 9.2% | 3.1% | 0.95 |
| realized_flat | 1996–2010 | 6.9% | 8.4% | 12.2% / 12.2% | 4.7% | 0.93 |
| realized_flat | 2011–2026 | 7.6% | 7.8% | 10.4% / 9.2% | 3.4% | 0.95 |

Some differences from PUT are expected:
* PUT writes SPX options (European, cash-settled), at a midday price.
* It has no costs.
* Yahoo's `^PUT` has bad prints, such as +30% and −33% on 2020-03-13 and 2020-03-16. That is why the comparison uses month-end values.

The naive model underpriced options by 1.5% a year in 1996–2010, the volatility risk premium it leaves out. After 2010 it happened to land closer to PUT's level, with a larger tracking error. Neither model is right in both halves.

## Results (vix_skew, tight costs)

| strategy | 1996–2010 CAGR | vol | max DD | excess Sharpe | 2011–2026 CAGR | vol | max DD | excess Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| buy-and-hold SPY | +6.1% | 21.5% | 55.2% | 0.24 | +13.9% | 17.0% | 33.7% | 0.76 |
| T-bills | +3.1% | 0.1% | 0% | — | +1.6% | 0.1% | 0% | — |
| covered call, ATM | +7.2% | 13.8% | 36.9% | 0.35 | +8.5% | 11.7% | 28.0% | 0.62 |
| covered call, 2% OTM | +7.5% | 16.2% | 40.9% | 0.34 | +10.9% | 13.7% | 31.5% | 0.71 |
| put write, ATM | +8.4% | 13.9% | 36.6% | 0.43 | +9.8% | 11.7% | 28.4% | 0.72 |
| put write, 2% OTM | +9.8% | 11.5% | 31.9% | 0.60 | +8.8% | 9.8% | 27.7% | 0.75 |

Crisis years:

| strategy | 2008 | 2020 | 2022 |
|---|---:|---:|---:|
| SPY | −36.8% | +18.4% | −18.2% |
| covered call, ATM | −26.2% | −1.8% | −7.3% |
| put write, ATM | −25.6% | −1.1% | −7.4% |

Writers lost about two thirds of SPY's 2008 drawdown, then missed the rebound. In 2020 they ended flat while SPY gained 18%.

## Other findings

* **Early assignment before SPY's ex-dividend dates costs the covered call about 1.2% a year** compared with the put write. Turning early exercise off removes the gap.
  * SPY goes ex-dividend around the quarterly expirations. An in-the-money call is rationally exercised the day before, so the writer gives up the dividend.
  * The strategy cannot see the assignment and sits in cash until the contract's expiration.
  * European SPX-style writes, like Cboe's BXM, avoid this.
* **Costs matter less than the volatility assumption.**
  * Going from frictionless to tight costs costs 0.3–0.4% a year; going to wide costs costs about 1.2%.
  * The choice of volatility model moves the same strategies by 2–4% a year.

## Caveats

* The synthetic contracts are fully liquid, and their marks are model prices, not trades.
* Skew, term structure (VIX is a 30-day measure; contracts here live about 20 sessions) and the VIX-to-ATM ratio are each a single assumption held fixed for 30 years.
* SPY dividends come from Yahoo, and the dividend yield is a trailing 12-month figure.
* One path, no parameter search beyond the single calibration, no deflation for the handful of variants.

## Next

Once the Massive key is in, fit the VIX-to-ATM ratio, its dependence on SKEW, and the skew itself on 2024–2026 SPY chains. Then rerun 2011–2026 as a test on data the fit never saw.
