//! Execution and accounting simulator.
//!
//! Timing is the single most common source of fake alpha, so it is fixed
//! here rather than left to callers:
//!
//! * A decision for step `t` is made after bar `t` closes (using only data
//!   known by then) and is executed at the **open of bar `t + 1`**.
//! * Every input that sizes or prices a trade at `t + 1` (`adv_shares`,
//!   `sigma`) must be computed from data up to bar `t`; the caller provides
//!   them already lagged (see the Python `MarketPanel`) and the simulator only
//!   reads row `t + 1` of those arrays, which by contract describe the state
//!   *before* that open.
//! * Corporate actions are applied at the open of their ex-date bar:
//!   dividends are credited (or, for shorts, debited) on the shares held at
//!   the previous close, then splits rescale share counts.
//! * A security that is listed but has no bar (a halt) cannot be traded; its
//!   position is frozen and valued at the last close. A security that is no
//!   longer listed is force-closed at its last close with a configurable
//!   haircut (no survivorship).
//! * A `NaN` target weight means "hold": shares are kept as they are and the
//!   weight drifts with the price, so discrete-trade strategies do not pay
//!   to rebalance every bar.
//!
//! Costs: commission (bps of notional), half-spread (bps), square-root market
//! impact `k * sigma * sqrt(|q| / ADV)`, a participation cap on `|q| / ADV`,
//! borrow fees on shorts, and financing on cash / margin.

use crate::error::{Error, Result};

/// `[T, S]` row-major market arrays.
#[derive(Clone, Debug)]
pub struct Market {
    pub periods: usize,
    pub securities: usize,
    pub open: Vec<f64>,
    pub close: Vec<f64>,
    /// Trailing average daily volume in shares known *before* the open of
    /// each bar.
    pub adv_shares: Vec<f64>,
    /// Trailing daily return volatility known before the open of each bar.
    pub sigma: Vec<f64>,
    /// Whether the security has a valid bar (listed and traded) at `t`.
    pub tradable: Vec<bool>,
    /// Whether the security is listed at `t` (tradable, or halted between
    /// its first and last bar). `close` must be the last valid close on
    /// listed-but-untradable bars.
    pub listed: Vec<bool>,
    /// Split ratio (new shares per old share) effective at the open of `t`.
    pub split: Vec<f64>,
    /// Cash dividend per pre-split share, paid at the open of its ex-date `t`.
    pub dividend: Vec<f64>,
}

impl Market {
    pub fn validate(&self) -> Result<()> {
        let n = self.periods * self.securities;
        let bad = |what: &str| Err(Error::Schema(format!("market: {what}")));
        if self.periods < 2 || self.securities == 0 {
            return bad("need at least 2 periods and 1 security");
        }
        for (name, len) in [
            ("open", self.open.len()),
            ("close", self.close.len()),
            ("adv_shares", self.adv_shares.len()),
            ("sigma", self.sigma.len()),
            ("tradable", self.tradable.len()),
            ("listed", self.listed.len()),
            ("split", self.split.len()),
            ("dividend", self.dividend.len()),
        ] {
            if len != n {
                return bad(&format!("{name} has {len} elements, expected {n}"));
            }
        }
        for i in 0..n {
            if self.tradable[i] {
                if !(self.open[i] > 0.0 && self.open[i].is_finite()) {
                    return bad(&format!(
                        "non-positive or non-finite open at flat index {i}"
                    ));
                }
                if !(self.close[i] > 0.0 && self.close[i].is_finite()) {
                    return bad(&format!(
                        "non-positive or non-finite close at flat index {i}"
                    ));
                }
            }
            if !(self.split[i] > 0.0 && self.split[i].is_finite()) {
                return bad(&format!(
                    "split ratio must be positive and finite (flat index {i})"
                ));
            }
            if !(self.dividend[i] >= 0.0 && self.dividend[i].is_finite()) {
                return bad(&format!(
                    "dividend must be non-negative and finite (flat index {i})"
                ));
            }
            if self.adv_shares[i].is_nan() || self.adv_shares[i] < 0.0 {
                return bad(&format!("adv_shares must be >= 0 (flat index {i})"));
            }
            if self.sigma[i].is_nan() || self.sigma[i] < 0.0 {
                return bad(&format!("sigma must be >= 0 (flat index {i})"));
            }
        }
        Ok(())
    }

    #[inline]
    fn at(&self, t: usize, i: usize) -> usize {
        t * self.securities + i
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Costs {
    pub commission_bps: f64,
    pub half_spread_bps: f64,
    /// Square-root impact coefficient `k`.
    pub impact_coef: f64,
    /// Max `|trade| / ADV` per bar; trades are clipped to it.
    pub max_participation: f64,
    /// Annual borrow fee on short market value.
    pub borrow_rate: f64,
    /// Annual rate earned on positive cash.
    pub cash_rate: f64,
    /// Annual rate paid on negative cash (margin).
    pub margin_rate: f64,
    /// Fraction of last close lost when a position is force-closed because
    /// its security stopped trading.
    pub delist_haircut: f64,
    pub periods_per_year: f64,
}

impl Default for Costs {
    /// Conservative defaults for liquid US equities at daily frequency.
    fn default() -> Self {
        Costs {
            commission_bps: 0.5,
            half_spread_bps: 2.0,
            impact_coef: 0.1,
            max_participation: 0.05,
            borrow_rate: 0.01,
            cash_rate: 0.0,
            margin_rate: 0.06,
            delist_haircut: 0.3,
            periods_per_year: 252.0,
        }
    }
}

impl Costs {
    fn validate(&self) -> Result<()> {
        let nonneg = [
            ("commission_bps", self.commission_bps),
            ("half_spread_bps", self.half_spread_bps),
            ("impact_coef", self.impact_coef),
            ("max_participation", self.max_participation),
            ("borrow_rate", self.borrow_rate),
            ("margin_rate", self.margin_rate),
        ];
        for (name, v) in nonneg {
            if !(v >= 0.0 && v.is_finite()) {
                return Err(Error::Schema(format!(
                    "costs.{name} must be finite and >= 0"
                )));
            }
        }
        if !self.cash_rate.is_finite() {
            return Err(Error::Schema("costs.cash_rate must be finite".into()));
        }
        if !(0.0..=1.0).contains(&self.delist_haircut) {
            return Err(Error::Schema(
                "costs.delist_haircut must be in [0, 1]".into(),
            ));
        }
        if !(self.periods_per_year > 0.0 && self.periods_per_year.is_finite()) {
            return Err(Error::Schema("costs.periods_per_year must be > 0".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Constraints {
    pub long_only: bool,
    /// Max `sum |w|` (1.0 = unlevered).
    pub max_gross: f64,
    /// Max `|w_i|`.
    pub max_weight: f64,
}

impl Default for Constraints {
    fn default() -> Self {
        Constraints {
            long_only: true,
            max_gross: 1.0,
            max_weight: 1.0,
        }
    }
}

/// What happened over one step (close `t` → close `t + 1`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepReport {
    /// Index of the bar the step ended on (`t + 1`).
    pub period: usize,
    pub equity_open: f64,
    pub equity: f64,
    /// `ln(equity_t+1 / equity_t)`; `-inf` on ruin.
    pub log_return: f64,
    pub gross: f64,
    pub net: f64,
    /// Traded notional / opening equity.
    pub turnover: f64,
    pub commission: f64,
    pub spread_cost: f64,
    pub impact_cost: f64,
    pub borrow_cost: f64,
    pub financing: f64,
    pub dividends: f64,
    pub delisted: usize,
    /// Securities whose trade was cut by the participation cap.
    pub clipped: usize,
    /// L1 distance between requested and constraint-projected weights.
    pub projection: f64,
    pub ruined: bool,
}

#[derive(Clone, Debug)]
pub struct Simulator {
    market: std::sync::Arc<Market>,
    costs: Costs,
    constraints: Constraints,
    t: usize,
    cash: f64,
    shares: Vec<f64>,
    equity: f64,
    ruined: bool,
    weights: Vec<f64>,
}

impl Simulator {
    pub fn new(
        market: std::sync::Arc<Market>,
        costs: Costs,
        constraints: Constraints,
        start: usize,
        capital: f64,
    ) -> Result<Self> {
        market.validate()?;
        costs.validate()?;
        if !(constraints.max_gross > 0.0 && constraints.max_gross.is_finite())
            || !(constraints.max_weight > 0.0 && constraints.max_weight.is_finite())
        {
            return Err(Error::Schema(
                "constraints must be positive and finite".into(),
            ));
        }
        if start + 1 >= market.periods {
            return Err(Error::Schema(format!(
                "start {start} leaves no step in {} periods",
                market.periods
            )));
        }
        if !(capital > 0.0 && capital.is_finite()) {
            return Err(Error::Schema("capital must be positive".into()));
        }
        let s = market.securities;
        Ok(Simulator {
            market,
            costs,
            constraints,
            t: start,
            cash: capital,
            shares: vec![0.0; s],
            equity: capital,
            ruined: false,
            weights: vec![0.0; s],
        })
    }

    pub fn period(&self) -> usize {
        self.t
    }

    pub fn done(&self) -> bool {
        self.ruined || self.t + 1 >= self.market.periods
    }

    pub fn equity(&self) -> f64 {
        self.equity
    }

    pub fn shares(&self) -> &[f64] {
        &self.shares
    }

    pub fn cash(&self) -> f64 {
        self.cash
    }

    /// Current weights at the last close.
    pub fn weights(&mut self) -> &[f64] {
        let m = &self.market;
        for i in 0..m.securities {
            let px = m.close[m.at(self.t, i)];
            self.weights[i] = if self.equity > 0.0 && self.shares[i] != 0.0 {
                self.shares[i] * px / self.equity
            } else {
                0.0
            };
        }
        &self.weights
    }

    /// Project `target` onto the constraints. `NaN` entries ("hold") and
    /// untradable securities are returned as `NaN`: they are not traded,
    /// and their gross exposure is part of `fixed_gross`.
    fn project(
        &self,
        target: &[f64],
        tradable_next: impl Fn(usize) -> bool,
        fixed_gross: f64,
    ) -> (Vec<f64>, f64) {
        let c = &self.constraints;
        let mut w: Vec<f64> = target
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                if x.is_nan() || !tradable_next(i) {
                    return f64::NAN;
                }
                let x = if c.long_only { x.max(0.0) } else { x };
                x.clamp(-c.max_weight, c.max_weight)
            })
            .collect();
        let gross: f64 = w.iter().filter(|x| !x.is_nan()).map(|x| x.abs()).sum();
        let budget = (c.max_gross - fixed_gross).max(0.0);
        if gross > budget {
            let scale = if gross > 0.0 { budget / gross } else { 0.0 };
            w.iter_mut()
                .filter(|x| !x.is_nan())
                .for_each(|x| *x *= scale);
        }
        // Requested-vs-executable distance; a held entry requests nothing.
        let dist = w
            .iter()
            .zip(target)
            .map(|(&a, &b)| match (a.is_nan(), b.is_nan()) {
                (_, true) => 0.0,
                (true, false) => b.abs(),
                (false, false) => (a - b).abs(),
            })
            .sum();
        (w, dist)
    }

    /// Execute `target` weights at the next open and mark to the next close.
    ///
    /// A `NaN` weight means "hold": the security's shares are left as they
    /// are (no trade, no cost), and its exposure counts against
    /// `max_gross` like a halted position. This is how strategies with
    /// discrete trades let weights drift instead of rebalancing every bar.
    pub fn step(&mut self, target: &[f64]) -> Result<StepReport> {
        if self.done() {
            return Err(Error::Schema("simulation is finished".into()));
        }
        let m = std::sync::Arc::clone(&self.market);
        let s = m.securities;
        if target.len() != s {
            return Err(Error::Dimension {
                context: "target weights".into(),
                expected: s,
                got: target.len(),
            });
        }
        if let Some(i) = target.iter().position(|x| x.is_infinite()) {
            return Err(Error::InvalidValue {
                context: "target weights".into(),
                index: i,
                value: target[i] as f32,
                reason: "infinite (use NaN to hold)",
            });
        }
        let (t0, t1) = (self.t, self.t + 1);
        let ppy = self.costs.periods_per_year;
        let mut r = StepReport {
            period: t1,
            ..StepReport::default()
        };

        // 1. Corporate actions at the open of t1.
        for i in 0..s {
            let q = self.shares[i];
            if q != 0.0 {
                let d = m.dividend[m.at(t1, i)];
                if d != 0.0 {
                    self.cash += q * d;
                    r.dividends += q * d;
                }
                self.shares[i] = q * m.split[m.at(t1, i)];
            }
        }
        // 2. Carry: borrow on shorts (valued at the last close), cash/margin.
        for i in 0..s {
            if self.shares[i] < 0.0 && m.listed[m.at(t0, i)] {
                // Split-adjust the last close to the post-split share count.
                let px = m.close[m.at(t0, i)] / m.split[m.at(t1, i)];
                r.borrow_cost += -self.shares[i] * px * self.costs.borrow_rate / ppy;
            }
        }
        r.financing = if self.cash >= 0.0 {
            self.cash * self.costs.cash_rate / ppy
        } else {
            self.cash * self.costs.margin_rate / ppy
        };
        self.cash += r.financing - r.borrow_cost;

        // 3. Forced exits for securities that are no longer listed.
        for i in 0..s {
            if self.shares[i] != 0.0 && !m.listed[m.at(t1, i)] {
                let last = m.close[m.at(t0, i)] / m.split[m.at(t1, i)];
                let h = self.costs.delist_haircut;
                let px = if self.shares[i] > 0.0 {
                    last * (1.0 - h)
                } else {
                    last * (1.0 + h)
                };
                self.cash += self.shares[i] * px;
                self.shares[i] = 0.0;
                r.delisted += 1;
            }
        }

        // 4. Opening equity; halted positions are valued at their last close.
        let open = |i: usize| {
            if m.tradable[m.at(t1, i)] {
                m.open[m.at(t1, i)]
            } else {
                m.close[m.at(t0, i)] / m.split[m.at(t1, i)]
            }
        };
        r.equity_open = self.cash
            + (0..s)
                .filter(|&i| self.shares[i] != 0.0)
                .map(|i| self.shares[i] * open(i))
                .sum::<f64>();
        let fixed_gross: f64 = (0..s)
            .filter(|&i| self.shares[i] != 0.0 && (!m.tradable[m.at(t1, i)] || target[i].is_nan()))
            .map(|i| (self.shares[i] * open(i)).abs())
            .sum::<f64>()
            / r.equity_open.max(f64::MIN_POSITIVE);
        if r.equity_open.is_nan() || r.equity_open <= 0.0 {
            return Ok(self.ruin(r));
        }

        // 5. Target shares, participation cap, fills.
        let (w, projection) = self.project(target, |i| m.tradable[m.at(t1, i)], fixed_gross);
        r.projection = projection;
        let bps = 1e-4;
        for (i, &wi) in w.iter().enumerate() {
            if wi.is_nan() {
                continue; // held or untradable
            }
            let px = open(i);
            let want = wi * r.equity_open / px;
            let mut q = want - self.shares[i];
            if q == 0.0 {
                continue;
            }
            let adv = m.adv_shares[m.at(t1, i)];
            let cap = self.costs.max_participation * adv;
            if q.abs() > cap {
                q = q.signum() * cap;
                r.clipped += 1;
            }
            if q == 0.0 {
                continue;
            }
            let notional = q.abs() * px;
            let impact_frac = if adv > 0.0 {
                self.costs.impact_coef * m.sigma[m.at(t1, i)] * (q.abs() / adv).sqrt()
            } else {
                0.0
            };
            let spread = notional * self.costs.half_spread_bps * bps;
            let impact = notional * impact_frac;
            let commission = notional * self.costs.commission_bps * bps;
            self.cash -= q * px + spread + impact + commission;
            self.shares[i] += q;
            r.turnover += notional;
            r.spread_cost += spread;
            r.impact_cost += impact;
            r.commission += commission;
        }
        r.turnover /= r.equity_open;

        // 6. Mark to the close of t1.
        let mut long = 0.0;
        let mut short = 0.0;
        for i in 0..s {
            if self.shares[i] != 0.0 {
                let v = self.shares[i] * m.close[m.at(t1, i)];
                if v > 0.0 { long += v } else { short += v }
            }
        }
        let equity = self.cash + long + short;
        self.t = t1;
        if equity.is_nan() || equity <= 0.0 {
            return Ok(self.ruin(r));
        }
        r.equity = equity;
        r.log_return = (equity / self.equity).ln();
        r.gross = (long - short) / equity;
        r.net = (long + short) / equity;
        self.equity = equity;
        Ok(r)
    }

    fn ruin(&mut self, mut r: StepReport) -> StepReport {
        self.ruined = true;
        self.t = r.period;
        self.shares.iter_mut().for_each(|q| *q = 0.0);
        self.cash = 0.0;
        self.equity = 0.0;
        r.equity = 0.0;
        r.log_return = f64::NEG_INFINITY;
        r.ruined = true;
        r
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn market(open: &[&[f64]], close: &[&[f64]]) -> Market {
        let t = open.len();
        let s = open[0].len();
        Market {
            periods: t,
            securities: s,
            open: open.concat(),
            close: close.concat(),
            adv_shares: vec![1e12; t * s],
            sigma: vec![0.02; t * s],
            tradable: vec![true; t * s],
            listed: vec![true; t * s],
            split: vec![1.0; t * s],
            dividend: vec![0.0; t * s],
        }
    }

    fn free() -> Costs {
        Costs {
            commission_bps: 0.0,
            half_spread_bps: 0.0,
            impact_coef: 0.0,
            borrow_rate: 0.0,
            margin_rate: 0.0,
            ..Costs::default()
        }
    }

    fn sim(m: Market, costs: Costs, c: Constraints) -> Simulator {
        Simulator::new(Arc::new(m), costs, c, 0, 1000.0).unwrap()
    }

    #[test]
    fn nan_holds_shares_and_drifts() {
        let o: &[&[f64]] = &[&[10.0, 10.0], &[10.0, 10.0], &[20.0, 10.0], &[20.0, 10.0]];
        let c: &[&[f64]] = &[&[10.0, 10.0], &[10.0, 10.0], &[20.0, 10.0], &[20.0, 10.0]];
        let costs = Costs {
            half_spread_bps: 10.0,
            ..free()
        };
        let mut s = sim(market(o, c), costs, Constraints::default());
        let r = s.step(&[0.5, 0.0]).unwrap();
        assert!(r.turnover > 0.0);
        let q = s.shares()[0];
        // Holding: no trade even though the weight drifted from 0.5 to 2/3.
        let r = s.step(&[f64::NAN, 0.0]).unwrap();
        assert_eq!(r.turnover, 0.0);
        assert_eq!(r.spread_cost, 0.0);
        assert_eq!(s.shares()[0], q);
        assert!((s.weights()[0] - 2.0 / 3.0).abs() < 1e-3);
        // A held position consumes the gross budget: the other leg is cut.
        let r = s.step(&[f64::NAN, 1.0]).unwrap();
        assert_eq!(s.shares()[0], q);
        assert!(r.projection > 0.0);
        // Costs are paid after sizing, so gross may exceed 1 by the cost.
        assert!(r.gross <= 1.0 + 1e-3, "{r:?}");
        assert!(s.done());
    }

    #[test]
    fn cash_is_flat_and_buy_and_hold_tracks_prices() {
        let m = market(&[&[10.0], &[11.0], &[12.0]], &[&[10.0], &[11.5], &[12.6]]);
        let mut s = sim(m.clone(), free(), Constraints::default());
        assert_eq!(s.step(&[0.0]).unwrap().log_return, 0.0);
        let mut s = sim(m, free(), Constraints::default());
        let r1 = s.step(&[1.0]).unwrap();
        // Bought at the next open (11), not the decision close (10).
        assert!((r1.equity - 1000.0 / 11.0 * 11.5).abs() < 1e-9);
        let r2 = s.step(&[1.0]).unwrap();
        assert!((r2.equity - 1000.0 / 11.0 * 12.6).abs() < 1e-9, "{r2:?}");
        assert!(s.done());
        assert!(s.step(&[1.0]).is_err());
    }

    #[test]
    fn splits_and_dividends_preserve_value() {
        // 2:1 split at t=2: prices halve, shares double; plus a 0.5 dividend
        // per pre-split share.
        let mut m = market(&[&[10.0], &[10.0], &[5.0]], &[&[10.0], &[10.0], &[5.0]]);
        m.split[2] = 2.0;
        m.dividend[2] = 0.5;
        let mut s = sim(m, free(), Constraints::default());
        s.step(&[1.0]).unwrap();
        let r = s.step(&[0.0]).unwrap();
        // 100 shares -> 200 at the open, sold at 5 => 1000, plus 50 dividend.
        assert!((r.dividends - 50.0).abs() < 1e-9);
        assert!((r.equity - 1050.0).abs() < 1e-9, "{r:?}");
    }

    #[test]
    fn costs_reduce_equity_and_are_accounted() {
        let m = market(&[&[10.0], &[10.0]], &[&[10.0], &[10.0]]);
        let costs = Costs {
            commission_bps: 1.0,
            half_spread_bps: 5.0,
            impact_coef: 0.0,
            ..free()
        };
        let mut s = sim(m, costs, Constraints::default());
        let r = s.step(&[1.0]).unwrap();
        let paid = r.commission + r.spread_cost;
        assert!((paid - 1000.0 * 6e-4).abs() < 1e-9);
        assert!((r.equity - (1000.0 - paid)).abs() < 1e-9);
        assert!((r.turnover - 1.0).abs() < 1e-12);
    }

    #[test]
    fn constraints_are_enforced() {
        let m = market(
            &[&[10.0, 10.0], &[10.0, 10.0]],
            &[&[10.0, 10.0], &[10.0, 10.0]],
        );
        let mut s = sim(m.clone(), free(), Constraints::default());
        let r = s.step(&[0.9, -0.5]).unwrap();
        assert!((r.projection - 0.5).abs() < 1e-12);
        assert!((r.gross - 0.9).abs() < 1e-12);
        let mut s = sim(
            m,
            free(),
            Constraints {
                long_only: false,
                max_gross: 1.0,
                max_weight: 0.6,
            },
        );
        let r = s.step(&[0.9, -0.9]).unwrap();
        assert!((r.gross - 1.0).abs() < 1e-9, "{r:?}");
        assert!(r.net.abs() < 1e-9);
    }

    #[test]
    fn participation_cap_clips() {
        let mut m = market(&[&[10.0], &[10.0]], &[&[10.0], &[10.0]]);
        m.adv_shares = vec![100.0, 100.0];
        let mut s = sim(m, free(), Constraints::default());
        let r = s.step(&[1.0]).unwrap();
        assert_eq!(r.clipped, 1);
        assert!((s.shares()[0] - 5.0).abs() < 1e-12); // 5% of 100 shares
    }

    #[test]
    fn delisting_forces_exit_with_haircut() {
        let mut m = market(&[&[10.0], &[10.0], &[10.0]], &[&[10.0], &[10.0], &[10.0]]);
        m.tradable[2] = false;
        m.listed[2] = false;
        let mut s = sim(m, free(), Constraints::default());
        s.step(&[1.0]).unwrap();
        let r = s.step(&[1.0]).unwrap();
        assert_eq!(r.delisted, 1);
        assert!((r.equity - 700.0).abs() < 1e-9);
    }

    #[test]
    fn halts_freeze_positions_without_haircut() {
        // Two securities; A is halted at t=2 (listed, no bar), resumes at t=3.
        let mut m = market(
            &[&[10.0, 10.0], &[10.0, 10.0], &[0.0, 10.0], &[12.0, 10.0]],
            &[&[10.0, 10.0], &[10.0, 10.0], &[10.0, 10.0], &[12.0, 10.0]],
        );
        m.tradable[2 * 2] = false;
        let mut s = sim(m, free(), Constraints::default());
        s.step(&[0.5, 0.5]).unwrap();
        // Asking to sell A during the halt does nothing; the gross budget
        // left for B is 0.5.
        let r = s.step(&[0.0, 1.0]).unwrap();
        assert_eq!(r.delisted, 0);
        assert!((s.shares()[0] - 50.0).abs() < 1e-9);
        assert!(
            (r.gross - 1.0).abs() < 1e-9 && (r.equity - 1000.0).abs() < 1e-9,
            "{r:?}"
        );
        let r = s.step(&[0.0, 1.0]).unwrap();
        assert!((s.shares()[0]).abs() < 1e-9);
        assert!((r.equity - 1100.0).abs() < 1e-9, "{r:?}");
    }

    #[test]
    fn ruin_is_terminal() {
        let m = market(&[&[10.0], &[10.0], &[30.0]], &[&[10.0], &[10.0], &[40.0]]);
        let mut s = sim(
            m,
            free(),
            Constraints {
                long_only: false,
                max_gross: 3.0,
                max_weight: 3.0,
            },
        );
        s.step(&[-3.0]).unwrap();
        let r = s.step(&[-3.0]).unwrap();
        assert!(r.ruined && r.log_return == f64::NEG_INFINITY);
        assert!(s.done());
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let m = market(&[&[10.0], &[10.0]], &[&[10.0], &[10.0]]);
        let mut s = sim(m.clone(), free(), Constraints::default());
        assert!(s.step(&[f64::INFINITY]).is_err());
        assert!(s.step(&[f64::NEG_INFINITY]).is_err());
        assert!(s.step(&[0.5, 0.5]).is_err());
        let mut bad = m;
        bad.open[1] = -1.0;
        assert!(Simulator::new(Arc::new(bad), free(), Constraints::default(), 0, 1.0).is_err());
    }
}
