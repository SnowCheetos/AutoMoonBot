# Stage 5: the graph transformer on a point-in-time S&P 500

**Result.** On a point-in-time S&P 500 the edge from the survivor-only run is gone:

- The graph transformer earns **+10.5% a year against +10.7% for equal weight**, with more volatility (26% against 21%) and a deeper drawdown (66% against 58%).
- Its factor alpha is **−1.0% a year (t −0.5)**; with the drawdown overlay it is −0.1% (t −0.06).

The 6–8% alpha (t ≈ 3) in [the survivor-only run](stage5-sp500-daily.md) came from survivorship bias, not from the model. Nothing about the model, its training or its hyperparameters changed between the two runs; only the universe and the classification did.

## What changed from the survivor-only run

**Membership.** The S&P 500's constituents on every date from 1998 come from the public change lists compiled in [fja05680/sp500](https://github.com/fja05680/sp500), read with `automoonbot.data.membership`:

- The current list (Nov 13, 2026) is appended as the last snapshot.
- A name counts as a member from the trading day its addition took effect, and stops on the day its removal did.
- Outside its membership a name's prices still feed the model's features as history, but the policy cannot hold it and the cross-sectional statistics leave it out.

**Former members' prices** come from Yahoo daily bars:

- 1,171 symbols were members at some point since 1998; 668 of them are no longer in the index.
- Yahoo returned a usable series for 274 of the 668. It keeps no history for most delisted tickers.
- 44 symbols whose current owner on Yahoo is not a common stock (an ETF or fund that took the ticker over) are dropped.
- Membership runs whose price series starts more than 21 bars after the run did are removed (`require_history`), for 91 symbols. These are almost all reused tickers, for example Citicorp's CCI before Crown Castle, Dollar General before its LBO and re-IPO, or a stub of a few bars that Yahoo returns for a ticker delisted this year.

The panel has 708 securities, against 503 in the survivor-only run.

**Classification.** GICS history is not freely available, so every company is classified by its SEC SIC code:

- the 4-digit code is the industry token and the 2-digit major group is the sector token;
- codes come from the SEC submissions API, with former members whose ticker no longer resolves matched by company name;
- only 5 names with prices could not be matched; they are excluded.

SIC is coarser and older than GICS, but it does not leak today's classification into the past the way the survivor run's GICS codes did.

**Coverage.** The fraction of index member-days for which the panel holds a tradable member bar:

| | 2006 | 2008 | 2010 | 2012 | 2014 | 2016 | 2018 | 2020 | 2022 | 2024 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| coverage | 55% | 60% | 64% | 68% | 71% | 76% | 81% | 87% | 91% | 95% | 98% |

The test period averages 77%. Almost all of the gap is names with no price series at all. The history check removes under one percentage point in any year.

## Results (Jul 2006 – Nov 2026, default costs)

Same walk-forward as before: 21 yearly folds, 21-day purge plus 5-day embargo, two causality audits per fold, $1M of capital. The training settings are identical to the survivor-only run. The weights rebuilt from the per-fold checkpoints reproduce the reported returns to 2×10⁻⁸ a day.

| strategy | log-growth / yr | vol | Sharpe (log) | max DD | turnover / yr | costs / yr |
|---|---|---|---|---|---|---|
| 5-day reversal (control) | **+12.5%** | 27.1% | 0.60 | 53.8% | 175 | 5.0% |
| high-vol quintile (control) | +11.8% | 32.5% | 0.53 | 75.9% | 5 | 0.2% |
| equal weight | +10.7% | 20.8% | 0.62 | 58.3% | 0.8 | 0.0% |
| buy and hold (equal, from 2006) | +10.6% | 20.1% | 0.63 | 56.5% | 0.3 | 0.0% |
| graph transformer | +10.5% | 26.3% | 0.53 | 66.4% | 114 | 3.3% |
| inverse volatility | +10.2% | 18.7% | 0.64 | 53.5% | 1.3 | 0.0% |
| cross-sectional momentum | +9.5% | 21.1% | 0.56 | 53.9% | 7 | 0.2% |
| low-vol quintile (control) | +9.3% | 14.6% | **0.71** | 41.1% | 5 | 0.1% |
| graph transformer + overlay | +6.5% | 16.9% | 0.47 | **41.2%** | 102 | 2.9% |

Even before costs, the model barely clears the passive portfolios:

- At tight costs the two model variants make +12.5% and +8.3% a year.
- At zero costs they make +13.8% and +9.4%.

That is +3.1% a year over equal weight before costs and the model takes more risk to get it, so the 3.3% a year it pays in costs is more than its whole edge.

### Survivor-only against point-in-time

| strategy | survivor-only | point-in-time | difference |
|---|---|---|---|
| graph transformer | +25.3% | +10.5% | −14.8% |
| graph transformer + overlay | +20.5% | +6.5% | −14.0% |
| high-vol quintile | +22.7% | +11.8% | −10.9% |
| 5-day reversal | +18.4% | +12.5% | −5.9% |
| cross-sectional momentum | +15.1% | +9.5% | −5.6% |
| equal weight | +14.8% | +10.7% | −4.1% |
| inverse volatility | +13.3% | +10.2% | −3.1% |
| low-vol quintile | +10.2% | +9.3% | −0.9% |

Survivorship inflated every strategy, in proportion to how much it bets on volatile names. Today's survivors include every volatile name that recovered from a deep drawdown and none that failed, so "buy the volatile names that fell" looks excellent in hindsight. The model learned exactly that, and lost the most: 14.8% a year, against 4.1% for equal weight.

### What the model holds now

- It is still fully invested on average (86–107% between the 10th and 90th percentiles), with about 41 effective names.
- Its average holding sits at the **64th percentile of 63-day volatility** (81st before), the **32nd of 12-1 momentum** (56th before), the 40th of 5-day return and the 51st of dollar volume.
- So it still leans toward volatile names, less than before, and now toward recent losers: a reversal bet.

**Single-factor regression** (daily log returns on equal weight):

| | beta | alpha / yr | t |
|---|---|---|---|
| graph transformer | 1.16 | −1.9% | −0.8 |
| + overlay | 0.67 | −0.7% | −0.3 |
| high-vol quintile | 1.47 | −3.8% | −1.6 |
| 5-day reversal | 1.19 | −0.3% | −0.1 |
| cross-sectional momentum | 0.90 | −0.1% | −0.0 |

**Factor regression.** The factors are equal weight, the high-minus-low volatility spread, and the reversal and momentum excess returns, all built on the same point-in-time universe.

| | alpha / yr | t | R² | loadings (EW, vol, reversal, momentum) |
|---|---|---|---|---|
| graph transformer | −1.0% | −0.5 | 0.88 | 0.96, 0.13, 0.25, −0.35 |
| + overlay | −0.1% | −0.1 | 0.70 | 0.60, 0.09, 0.03, 0.00 |

The model is now mostly the market, with a reversal tilt and a short momentum tilt, and nothing left over.

**By year** (log-growth against equal weight):

- **The model's whole edge is one year.** In the 2009 rebound it makes +73% against +38% for equal weight. Over the other 20 years it trails equal weight by about 2% a year.
- Its other wins are small: 2006 (+16% against +11%), 2010, 2012, 2016, 2019, 2021 and 2023.
- Its worst years are 2007 (−12% against +3%), 2011 (−9% against +2%), 2015 (−13% against −2%), 2017 (+9% against +19%) and 2024 (−7% against +12%).
- It lags equal weight in each of the last three years.

**The overlay** cuts 2008 from −48% to −22%, but it is still de-risked in the rebounds that follow: it makes +43% in 2009 and −13% in 2020, where the model alone makes +73% and +10%. That costs it 4% a year over the full period.

## Reading

1. **No edge.** On this universe the graph transformer does not beat equal weight, its factor alpha is indistinguishable from zero, and its costs exceed its pre-cost edge. The survivor-only result was an artefact of the universe.
2. **The remaining holes do not rescue it.** Coverage is 55–70% of member-days in 2006–2013, and the missing names are the ones that later failed or were acquired. These are the names survivorship hides, so this universe is still partly survivorship-biased. The direction of the remaining bias should still favour a model that buys volatile losers, so filling the holes (with delisting returns) is more likely to make the result worse than better.
3. **Classification confound.** GICS changed to SIC at the same time as the universe changed, so part of the model's loss could come from a weaker graph. Two things argue against that being most of it:
   - The controls, which use no graph at all, lost most of their excess return too (high-vol quintile: +7.9% a year over equal weight before, +1.2% now).
   - The model's factor alpha fell from 5.7% to −1.0%.

   A survivor-only run with SIC classification would separate the two effects exactly.
4. **What this says about the approach.** Price and volume features alone, with an industry and sector graph, do not carry a signal the model can exploit after costs on large-cap US stocks. That is the expected answer for the most efficient equity market there is. The pipeline itself held up: point-in-time membership, causality audits and the reproduction check all behaved. The graph has to carry information the price series does not, which is the job of stages 6 and 7 (relation edges from filings and news, supplier–customer spillovers), not of a bigger model on the same inputs.

## Reproducing

```sh
python experiments/stage5_sp500_daily.py --store data/d1 --universe sp500.csv \
    --membership sp500_history.csv --sic sic.json --instruments former_yahoo.json \
    --out results/s5pit --bf16 --threads 3 \
    --train '{"epochs": 20, "warm_epochs": 6, "steps_per_epoch": 60, "patience": 4}'
python experiments/stage5_analysis.py <same universe flags> --run results/s5pit
```

- `sp500_history.csv` is the fja05680/sp500 *Historical Components & Changes* file (`date,tickers`).
- `sic.json` holds SIC codes from the SEC submissions API.
- `former_yahoo.json` holds Yahoo's instrument type per former member.
- `universe.json` in the run directory records the excluded symbols, coverage before and after the history check, and the symbols the check removed.

Training took 6.2 hours on 3 CPU cores.
