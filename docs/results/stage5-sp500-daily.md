# Stage 5: graph transformer on S&P 500 daily bars

**Result.** Out of sample, the graph transformer beats every baseline on the same universe. After controlling for the obvious factor tilts, the alpha is about 6–8% a year (t ≈ 3).

**This is not yet evidence of a real edge.** The universe is the *current* S&P 500, so it is survivorship-biased. The model leans hard into high-volatility names, and that is exactly the tilt survivorship flatters. The test that decides it is a point-in-time universe that includes names later removed or delisted.

## Setup

**Universe**
- 503 current S&P 500 constituents, with their current GICS sector and sub-industry.
- Survivorship-biased in two ways:
  - Membership is today's: no delisted or removed names, and companies appear before they joined the index.
  - Classification is today's: each company's GICS codes are asserted from its first bar.

**Data**
- Yahoo daily bars from 1998 (7,262 bars), un-adjusted to as-traded prices.
- The simulator applies splits and dividends.

**Walk-forward**
- 21 yearly test windows, mid-2006 to Nov 2026, forming one continuous path.
- Expanding training sets, with labels purged by the 21-day horizon plus a 5-day embargo.
- Two random causality audits per fold for every strategy.
- The model's weights were rebuilt from the per-fold checkpoints and reproduce the reported returns exactly.

**Execution**
- Next-open fills, at $1M of capital.
- Default costs: 0.5 bp commission, 2 bp half-spread, square-root impact.
- Every strategy is also scored at tight costs and at zero costs, from the same fit.

**Model** (`automoonbot.models`)
- **Inputs:** 8 causal price channels, robust-standardized across the cross-section each day.
- **Temporal encoder:** PatchTST-style, over 64 days.
- **Graph layers:** three GPS-style layers over company, industry and sector tokens. Each does typed message passing along the graph edges known that day, then global attention.
- **Heads:** chained Student-t heads for the 1, 5 and 21-day returns (multi-token prediction style).
- **Policy:** a count-invariant exposure head and a softmax allocation over names, capped at 5% per name.

**Training**
- The objective is the realized cost-adjusted portfolio log-growth (5 bp per unit of turnover), minus 0.01 × daily CVaR95, plus 0.1 × the auxiliary NLL. It is differentiable end to end, so no critic is needed.
- Each batch is a random 192-name sub-universe.
- Early stopping on a purged 1-year validation tail, with warm starts across folds.
- Hyperparameters were set before the run and not tuned on these results.
- A one-year pilot (2006, the first fold) ran first, with shorter training.

**Inference**
- A 0.2% no-trade band.
- An optional drawdown overlay: exposure falls linearly from 100% at a 10% drawdown to 25% at a 25% drawdown. Its settings were fixed before the run.

**Compute:** about 5.6 hours on 2 CPU cores, bf16 autocast.

## Results (Jul 2006 – Nov 2026, default costs)

| strategy | log-growth / yr | vol | Sharpe (log) | max DD | turnover / yr | costs / yr |
|---|---|---|---|---|---|---|
| graph transformer | **+25.3%** | 29.6% | 1.00 | 59.0% | 129 | 4.7% |
| graph transformer + overlay | +20.5% | 22.0% | **1.04** | **36.0%** | 113 | 3.7% |
| high-vol quintile (control) | +22.7% | 31.0% | 0.89 | 65.5% | 5 | 0.2% |
| 5-day reversal (control) | +18.4% | 26.1% | 0.84 | 51.0% | 175 | 5.3% |
| cross-sectional momentum | +15.1% | 22.1% | 0.79 | 56.5% | 7 | 0.2% |
| equal weight | +14.8% | 20.4% | 0.83 | 52.1% | 0.7 | 0.0% |
| buy and hold (equal, from 2006) | +14.8% | 19.9% | 0.84 | 51.2% | 0.2 | 0.0% |
| inverse volatility | +13.3% | 18.5% | 0.72 | 49.1% | 1.3 | 0.0% |
| low-vol quintile (control) | +10.2% | 14.6% | 0.77 | 42.0% | 5 | 0.1% |

At zero costs the two model variants reach +30.0% and +24.2% a year.

The deflated Sharpe ratio is about 1.0 for every positive strategy. All of them sit far above zero, so here it does not separate them.

### Where the excess return comes from

**Holdings**
- Fully invested on average (the exposure head moves between 95% and 106%; the overlay is what cuts risk).
- About 47 effective names.
- Average holding sits at the **81st percentile of 63-day volatility**, the 56th of 12-1 momentum, the 43rd of 5-day return and the 47th of dollar volume.

**Single-factor regression** (daily log returns on equal weight):

| | beta | alpha / yr | t |
|---|---|---|---|
| graph transformer | 1.27 | 6.5% | 2.1 |
| + overlay | 0.81 | 8.4% | 2.6 |
| high-vol quintile | 1.43 | 1.4% | 0.6 |
| 5-day reversal | 1.18 | 0.9% | 0.4 |

**Factor regression.** The factors are equal weight, the high-minus-low volatility spread, and the reversal and momentum excess returns. Every factor is built on the same universe, so each control is equally flattered by survivorship.

| | alpha / yr | t | R² | loadings (EW, vol, reversal, momentum) |
|---|---|---|---|---|
| graph transformer | 5.7% | 2.8 | 0.90 | 0.80, 0.54, 0.24, 0.40 |
| + overlay | 7.8% | 3.3 | 0.76 | 0.50, 0.43, −0.04, 0.62 |

**By year**
- The model's excess return is lumpy.
- Big wins come in high-volatility rebounds and rallies: 2009 (+104% vs +39%), 2020 (+66% vs +22%), and 2023–2026.
- It lags in 2014–2016 and 2022 (−28% vs −11%).
- The overlay roughly halves the 2008 and 2022 damage.

## Reading

1. **What is solid.** On identical data, the model beats each simple rule, including the controls built to copy its tilts, and the causality audit holds bit for bit. The pipeline does what it should.
2. **Survivorship.** Survivorship bias is the leading alternative explanation:
   - In a universe of today's survivors, every deep drawdown in the sample eventually recovered.
   - So a flexible model learns "buy volatile names after they fall".
   - The linear controls take out part of this effect, but not a nonlinear version of it.
   - The model's biggest years are exactly the rebounds where survivorship helps most.
3. **Turnover.** 113–130 turns a year costs 4–5% a year even at $1M. At realistic institutional size, impact would eat more. The training cost of 5 bp is the right order, but a capacity study is needed before any live use.
4. **Next experiment.** Re-run unchanged on a point-in-time S&P 500:
   - historical membership;
   - price histories for removed and delisted names, with delisting returns.
   
   If the factor alpha survives that, it is worth taking seriously. The other planned ablations should follow it: the heterogeneous GNN, no graph (company tokens only), and multiple seeds.

Scripts: `experiments/stage5_sp500_daily.py` (the run) and `experiments/stage5_analysis.py` (the diagnostics).
