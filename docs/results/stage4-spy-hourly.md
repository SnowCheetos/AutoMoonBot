# Stage 4: PGPerceptron out of sample (SPY hourly)

PGPerceptron was the old project's best model. These are its results
re-measured out of sample, with the same harness every new model goes through.

## Setup

**Data**
- SPY regular-session hourly bars from Yahoo: Jul 2024 to Nov 2026, 4,080 bars.
- The bars are converted back to as-traded prices; splits and dividends are applied by the simulator.

**Protocol**
- The original setting is kept: retrain every 50 bars on the last 1,000 bars.
- That gives 82 walk-forward folds whose test windows join into one continuous out-of-sample path.
- Every fit sees only a panel truncated before its test window.
- Random causality audits run on every fold.

**Execution**
- Fills at the next bar's open.
- Three cost settings:
  - default: 0.5 bp commission and 2 bp half-spread, plus square-root impact
  - tight: 0.2 bp commission and 0.5 bp half-spread
  - frictionless: no costs

**Variants**
- *legacy* is the original algorithm as written, including its four bugs:
  - the position input was always Cash
  - rewards were credited to the next action
  - the best-weights revert was a no-op
  - the live manager ran at leverage 0
- *repaired* has those bugs fixed.
- Model selection is either on the training data (*in_sample*) or on a held-out tail (*validation*).

**Seeds:** three (0, 1, 2). Rule baselines are deterministic.

## Results (log-growth over the whole path, default costs)

| strategy | seed 0 | seed 1 | seed 2 | vol (ann.) | max DD | turnover / yr |
|---|---|---|---|---|---|---|
| buy and hold SPY | +31.7% | | | 16.0% | 19.2% | 0 |
| 50% SPY, held | +16.9% | | | 8.2% | 10.0% | 0 |
| trend following | +12.8% | | | 12.3% | 10.3% | 10 |
| PGP repaired, in-sample selection | +1.3% | +11.0% | +2.8% | 7.7–9.1% | 8.8–11.7% | 116–132 |
| PGP repaired, validation selection | −2.1% | +9.8% | −2.0% | 7.1–10.1% | 9.6–16.6% | 116–135 |
| PGP legacy | −5.2% | +14.0% | −12.4% | 8.3–11.3% | 7.6–18.0% | 81–136 |

Without costs, repaired PGP reaches +8.2%, +18.8% and +10.3% for seeds 0, 1 and 2.
Costs are about 3% a year because the strategy turns over 80–135 times a year.

## Conclusions

- **No variant or seed beats an exposure-matched passive position.** The best run, legacy with seed 1, makes +14.0% at 8.5% volatility. Simply holding 50% SPY makes +16.9% at 8.2%.
- **The spread between seeds is larger than any difference between variants.** Legacy ranges from −12.4% to +14.0% depending only on the random seed. Most of what the model "learns" is noise.
- **Fixing the bugs makes results less bad and more stable**, but adds no edge.
- **PGPerceptron is retired as a candidate.** It stays in the codebase as a baseline and as a regression test for the harness.

The script is `experiments/stage4_spy_hourly.py`.
