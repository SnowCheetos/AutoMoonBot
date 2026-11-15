//! Execution and accounting simulator.
//!
//! Timing is the single most common source of fake alpha, so it is fixed
//! here rather than left to callers:
//!
//! * A decision for step `t` is made after bar `t` closes (using only data
//!   known by then) and is executed at the **open of bar `t + 1`**.
//! * Every input that sizes or prices a trade at `t + 1` (`adv_shares`,
//!   `sigma`, `half_spread`) must be computed from data up to bar `t`; the
//!   caller provides them already lagged (see the Python `MarketPanel`) and
//!   the simulator only reads row `t + 1` of those arrays, which by contract
//!   describe the state *before* that open.
//! * Corporate actions are applied at the open of their ex-date bar:
//!   dividends are credited (or, for shorts, debited) on the shares held at
//!   the previous close, then splits rescale share counts and adjust the
//!   terms of options on the stock.
//! * A security that is listed but has no bar (a halt) cannot be traded; its
//!   position is frozen and valued at the last close. A security that is no
//!   longer listed is force-closed at its last close with a configurable
//!   haircut (no survivorship).
//! * A `NaN` target means "hold": the position is kept as it is and its
//!   weight drifts with the price, so discrete-trade strategies do not pay
//!   to rebalance every bar.
//!
//! Costs: commission (bps of notional for stock, per contract for options
//! and futures), half-spread (bps, or an absolute per-bar spread when the
//! market provides one), square-root market impact `k * sigma * sqrt(|q| /
//! ADV)`, a participation cap on `|q| / ADV`, borrow fees on short stock,
//! and financing on cash / margin.
//!
//! # Derivatives and margin
//!
//! Columns can be options and futures as well as stock (see
//! [`instrument`]). Positions are counts of units (shares or contracts);
//! targets are either weights of equity ([`Simulator::step`]) or units
//! ([`Simulator::step_units`]). The weight of a derivative is its notional
//! over equity: a future's contract value, and for an option the value of
//! the underlying it delivers (`deliverable` units at the underlying's
//! price), not its premium. So writing calls on a stock position of weight
//! 1 is a call weight of -1, whatever the premium, and both legs are sized
//! from the same opening price of the stock.
//!
//! * **Futures** are marked to market: gains and losses move to cash at the
//!   open (before trading) and at the close of every bar, and a position
//!   adds nothing to equity beyond that. At expiry the position is settled
//!   at the final settlement price: its own close on the last trading bar,
//!   or the underlying's open on the next bar.
//! * **Options** are bought and sold for their premium. After the close of
//!   the last trading bar an option at least `exercise_threshold` in the
//!   money is exercised (long) or assigned (short); physical settlement
//!   delivers the underlying at the strike, cash settlement pays the
//!   intrinsic value at the settlement price. Others expire worthless.
//! * **Early exercise** (American options, when enabled) follows the
//!   rational holder: a call is exercised after the close before an
//!   ex-dividend bar when the dividend exceeds its time value, and a put
//!   when its time value is at most `exercise_threshold`. It applies to
//!   long positions (exercised) and short ones (assigned) alike.
//! * **Margin** ([`margin`]): new trades are scaled back until the book
//!   meets the initial requirement (a trade that does not raise an
//!   existing deficiency is allowed). At every close the maintenance
//!   requirement is checked; if equity is short of it, every tradable
//!   position is cut by the same fraction at the next open until the book
//!   meets the requirement again, and the strategy's target for that bar
//!   is not executed.

pub mod instrument;
pub mod margin;
pub mod pricing;

use std::sync::Arc;

use crate::error::{Error, Result};
use instrument::{Instrument, Kind, Settlement, SettlementPrice, Style};
use margin::{Groups, MarginModel, Marks, requirement};
use pricing::Right;

/// `[T, S]` row-major market arrays.
#[derive(Clone, Debug)]
pub struct Market {
    pub periods: usize,
    pub securities: usize,
    pub open: Vec<f64>,
    pub close: Vec<f64>,
    /// Trailing average daily volume in units (shares or contracts) known
    /// *before* the open of each bar.
    pub adv_shares: Vec<f64>,
    /// Trailing daily return volatility known before the open of each bar.
    pub sigma: Vec<f64>,
    /// Whether the security has a valid bar (listed and traded) at `t`.
    pub tradable: Vec<bool>,
    /// Whether the security is listed at `t` (tradable, or halted between
    /// its first and last bar). `close` must be the last valid close on
    /// listed-but-untradable bars.
    pub listed: Vec<bool>,
    /// Split ratio (new units per old unit) effective at the open of `t`.
    /// For an option it is the ratio of its contract count.
    pub split: Vec<f64>,
    /// Cash dividend per pre-split share, paid at the open of its ex-date `t`.
    pub dividend: Vec<f64>,
    /// What each column is; empty means every column is a stock.
    pub instruments: Vec<Instrument>,
    /// Optional absolute half-spread per unit of price, known before the
    /// open of each bar; `NaN` entries fall back to the bps defaults.
    pub half_spread: Option<Vec<f64>>,
}

impl Market {
    /// A market of stocks only, from the eight standard arrays.
    #[allow(clippy::too_many_arguments)]
    pub fn equities(
        periods: usize,
        securities: usize,
        open: Vec<f64>,
        close: Vec<f64>,
        adv_shares: Vec<f64>,
        sigma: Vec<f64>,
        tradable: Vec<bool>,
        listed: Vec<bool>,
        split: Vec<f64>,
        dividend: Vec<f64>,
    ) -> Self {
        Market {
            periods,
            securities,
            open,
            close,
            adv_shares,
            sigma,
            tradable,
            listed,
            split,
            dividend,
            instruments: Vec::new(),
            half_spread: None,
        }
    }

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
            ("half_spread", self.half_spread.as_ref().map_or(n, Vec::len)),
        ] {
            if len != n {
                return bad(&format!("{name} has {len} elements, expected {n}"));
            }
        }
        if !self.instruments.is_empty() && self.instruments.len() != self.securities {
            return bad(&format!(
                "{} instruments for {} securities",
                self.instruments.len(),
                self.securities
            ));
        }
        for (i, inst) in self.instruments.iter().enumerate() {
            inst.validate(i, &self.instruments)?;
        }
        let pos = |x: f64| x > 0.0 && x.is_finite();
        for k in 0..n {
            let (t, i) = (k / self.securities, k % self.securities);
            let kind = self.instrument(i).kind;
            if let Kind::Index { .. } = kind {
                if self.tradable[k] {
                    return bad(&format!("index column {i} cannot be tradable (t={t})"));
                }
                // Settlement and margin read an index's levels whenever it
                // is listed.
                if self.listed[k] && !(pos(self.open[k]) && pos(self.close[k])) {
                    return bad(&format!(
                        "index column {i} needs positive levels when listed (t={t})"
                    ));
                }
            }
            if let Some(e) = self.instrument(i).expiry()
                && t > e
                && self.listed[k]
            {
                return bad(&format!(
                    "column {i} is listed at {t}, after its expiry {e}"
                ));
            }
            if !matches!(kind, Kind::Equity) && self.dividend[k] != 0.0 {
                return bad(&format!("only stocks pay dividends (column {i}, t={t})"));
            }
            if matches!(kind, Kind::Future(_) | Kind::Index { .. }) && self.split[k] != 1.0 {
                return bad(&format!(
                    "futures and indices cannot split (column {i}, t={t})"
                ));
            }
            if self.tradable[k] {
                if !pos(self.open[k]) {
                    return bad(&format!(
                        "non-positive or non-finite open at flat index {k}"
                    ));
                }
                if !pos(self.close[k]) {
                    return bad(&format!(
                        "non-positive or non-finite close at flat index {k}"
                    ));
                }
            }
            if !pos(self.split[k]) {
                return bad(&format!(
                    "split ratio must be positive and finite (flat index {k})"
                ));
            }
            if !(self.dividend[k] >= 0.0 && self.dividend[k].is_finite()) {
                return bad(&format!(
                    "dividend must be non-negative and finite (flat index {k})"
                ));
            }
            if self.adv_shares[k].is_nan() || self.adv_shares[k] < 0.0 {
                return bad(&format!("adv_shares must be >= 0 (flat index {k})"));
            }
            if self.sigma[k].is_nan() || self.sigma[k] < 0.0 {
                return bad(&format!("sigma must be >= 0 (flat index {k})"));
            }
            if let Some(h) = &self.half_spread
                && !(h[k].is_nan() || (h[k] >= 0.0 && h[k].is_finite()))
            {
                return bad(&format!("half_spread must be NaN or >= 0 (flat index {k})"));
            }
        }
        Ok(())
    }

    #[inline]
    fn at(&self, t: usize, i: usize) -> usize {
        t * self.securities + i
    }

    #[inline]
    pub fn instrument(&self, i: usize) -> &Instrument {
        self.instruments.get(i).unwrap_or(&Instrument::EQUITY)
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
    /// Annual borrow fee on short stock market value.
    pub borrow_rate: f64,
    /// Annual rate earned on positive cash.
    pub cash_rate: f64,
    /// Annual rate paid on negative cash (margin).
    pub margin_rate: f64,
    /// Fraction of last close lost when a position is force-closed because
    /// its security stopped trading.
    pub delist_haircut: f64,
    pub periods_per_year: f64,
    /// Commission per option contract.
    pub option_commission: f64,
    /// Commission per future contract (including exchange fees).
    pub future_commission: f64,
    /// Option half-spread: the larger of these bps of premium and this
    /// absolute amount per unit of price.
    pub option_half_spread_bps: f64,
    pub option_min_half_spread: f64,
}

impl Default for Costs {
    /// Conservative defaults for liquid US equities and listed derivatives
    /// at daily frequency.
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
            option_commission: 0.65,
            future_commission: 2.25,
            option_half_spread_bps: 100.0,
            option_min_half_spread: 0.005,
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
            ("option_commission", self.option_commission),
            ("future_commission", self.future_commission),
            ("option_half_spread_bps", self.option_half_spread_bps),
            ("option_min_half_spread", self.option_min_half_spread),
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
    /// Max `sum |w|` (1.0 = unlevered), with derivatives at their notional
    /// (see the module docs): a covered call has a gross of 2.
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

/// Exercise and margin rules.
#[derive(Clone, Debug, PartialEq)]
pub struct Rules {
    pub margin: MarginModel,
    /// Exercise American options early when a rational holder would.
    pub early_exercise: bool,
    /// Minimum in-the-money amount per delivered unit for exercise at
    /// expiry, and the time value at or below which a put is exercised
    /// early (OCC: $0.01).
    pub exercise_threshold: f64,
    /// Continuously compounded rate for option revaluation in margin
    /// scenarios.
    pub rate: f64,
}

impl Default for Rules {
    fn default() -> Self {
        Rules {
            margin: MarginModel::None,
            early_exercise: true,
            exercise_threshold: 0.01,
            rate: 0.0,
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
    /// Cash moved by futures marking to market over the step.
    pub variation_margin: f64,
    pub delisted: usize,
    /// Securities whose trade was cut by the participation cap.
    pub clipped: usize,
    /// L1 distance between requested and constraint-projected weights.
    pub projection: f64,
    /// Long option positions exercised and short ones assigned (at expiry
    /// or early), and positions that expired worthless or were cash-settled.
    pub exercised: usize,
    pub assigned: usize,
    pub expired: usize,
    /// Maintenance requirement at the close (0 without a margin model).
    pub margin_requirement: f64,
    /// Equity was below the maintenance requirement at the close; the book
    /// is cut at the next open.
    pub margin_call: bool,
    /// Positions cut at the open to meet a margin call.
    pub liquidated: usize,
    /// Fraction of the requested trades executed after the initial-margin
    /// check (1: all of them).
    pub margin_scale: f64,
    pub ruined: bool,
}

#[derive(Clone, Debug)]
pub struct Simulator {
    market: Arc<Market>,
    costs: Costs,
    constraints: Constraints,
    rules: Rules,
    groups: Groups,
    instruments: Vec<Instrument>,
    /// Columns that are options / futures.
    options: Vec<usize>,
    futures: Vec<usize>,
    t: usize,
    cash: f64,
    shares: Vec<f64>,
    /// Current option terms (adjusted by splits).
    strike: Vec<f64>,
    deliverable: Vec<f64>,
    /// Last settlement price of each future.
    mark: Vec<f64>,
    equity: f64,
    margin_call: bool,
    ruined: bool,
    weights: Vec<f64>,
}

/// Whether a step's target is weights of equity or units.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Target {
    Weights,
    Units,
}

impl Simulator {
    pub fn new(
        market: Arc<Market>,
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
        let instruments: Vec<Instrument> = (0..s).map(|i| *market.instrument(i)).collect();
        let mut strike = vec![0.0; s];
        let mut deliverable = vec![0.0; s];
        for (i, inst) in instruments.iter().enumerate() {
            if let Kind::Option(o) = inst.kind {
                strike[i] = o.strike;
                deliverable[i] = o.deliverable;
            }
        }
        let options = (0..s).filter(|&i| instruments[i].is_option()).collect();
        let futures = (0..s).filter(|&i| instruments[i].is_future()).collect();
        Ok(Simulator {
            groups: Groups::new(&instruments),
            instruments,
            options,
            futures,
            market,
            costs,
            constraints,
            rules: Rules::default(),
            t: start,
            cash: capital,
            shares: vec![0.0; s],
            strike,
            deliverable,
            mark: vec![f64::NAN; s],
            equity: capital,
            margin_call: false,
            ruined: false,
            weights: vec![0.0; s],
        })
    }

    /// Replace the exercise and margin rules (before the first step).
    pub fn with_rules(mut self, rules: Rules) -> Result<Self> {
        rules.margin.validate()?;
        if !(rules.exercise_threshold >= 0.0 && rules.exercise_threshold.is_finite())
            || !rules.rate.is_finite()
        {
            return Err(Error::Schema(
                "rules: invalid exercise threshold or rate".into(),
            ));
        }
        self.rules = rules;
        Ok(self)
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

    /// Units held: shares, or contracts.
    pub fn shares(&self) -> &[f64] {
        &self.shares
    }

    pub fn cash(&self) -> f64 {
        self.cash
    }

    /// Current option strikes and deliverables (after split adjustments).
    pub fn option_terms(&self, i: usize) -> Option<(f64, f64)> {
        self.instruments[i]
            .is_option()
            .then(|| (self.strike[i], self.deliverable[i]))
    }

    /// Current weights at the last close (derivatives at their notional).
    pub fn weights(&mut self) -> &[f64] {
        let m = self.market.clone();
        let t = self.t;
        for i in 0..m.securities {
            self.weights[i] = if self.equity > 0.0 && self.shares[i] != 0.0 {
                self.shares[i] * self.unit_exposure(i, |j| m.close[m.at(t, j)]) / self.equity
            } else {
                0.0
            };
        }
        &self.weights
    }

    /// The exposure of one unit of `i`, the basis of its weight, given each
    /// column's price: a share's or future's contract value, and for an
    /// option the value of the underlying it delivers.
    fn unit_exposure(&self, i: usize, price: impl Fn(usize) -> f64) -> f64 {
        let mult = self.instruments[i].multiplier;
        match self.instruments[i].kind {
            Kind::Option(o) => price(o.underlying) * self.ratio(i) * mult,
            _ => price(i) * mult,
        }
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
    /// A `NaN` weight means "hold": the security's position is left as it
    /// is (no trade, no cost), and its exposure counts against `max_gross`
    /// like a halted position. This is how strategies with discrete trades
    /// let weights drift instead of rebalancing every bar.
    pub fn step(&mut self, target: &[f64]) -> Result<StepReport> {
        self.step_with(target, Target::Weights)
    }

    /// As [`step`](Self::step), with the target in units (shares or
    /// contracts; `NaN`: hold). The constraints apply to the weights the
    /// units imply at the next open.
    pub fn step_units(&mut self, target: &[f64]) -> Result<StepReport> {
        self.step_with(target, Target::Units)
    }

    #[inline]
    fn is_index(&self, i: usize) -> bool {
        matches!(self.instruments[i].kind, Kind::Index { .. })
    }

    /// Option price per quote unit for a spot `s` of its underlying, at
    /// intrinsic value (`ratio` delivered units per quote unit).
    fn intrinsic(&self, i: usize, spot: f64) -> f64 {
        let Kind::Option(o) = self.instruments[i].kind else {
            return 0.0;
        };
        let ratio = self.ratio(i);
        o.right.intrinsic(spot * ratio, self.strike[i])
    }

    /// Underlying value delivered per unit of the option's quoted price.
    fn ratio(&self, i: usize) -> f64 {
        match self.instruments[i].kind {
            Kind::Option(o) if o.settlement == Settlement::Physical => {
                self.deliverable[i] * self.instruments[o.underlying].multiplier
                    / self.instruments[i].multiplier
            }
            _ => 1.0,
        }
    }

    /// Settle option `i`'s whole position by exercise against `spot`
    /// (cash settlement at intrinsic value, or physical delivery at the
    /// strike). `mark_u` is the current settlement price of a futures
    /// underlying.
    fn exercise(&mut self, i: usize, spot: f64, r: &mut StepReport) {
        let q = self.shares[i];
        let Kind::Option(o) = self.instruments[i].kind else {
            return;
        };
        if q == 0.0 {
            return;
        }
        let mult = self.instruments[i].multiplier;
        match o.settlement {
            Settlement::Cash(_) => self.cash += q * mult * self.intrinsic(i, spot),
            Settlement::Physical => {
                let u = o.underlying;
                let sign = if o.right == Right::Call { 1.0 } else { -1.0 };
                // Delivered units, and the aggregate exercise price per
                // delivered unit.
                let du = sign * q * self.deliverable[i];
                let k_unit =
                    self.strike[i] * mult / (self.deliverable[i] * self.instruments[u].multiplier);
                let umult = self.instruments[u].multiplier;
                if self.instruments[u].is_future() {
                    // The new futures are struck at `k_unit` and marked at
                    // the current settlement price at once.
                    let vm = du * umult * (self.mark[u] - k_unit);
                    self.cash += vm;
                    r.variation_margin += vm;
                } else {
                    self.cash -= du * umult * k_unit;
                }
                self.shares[u] += du;
            }
        }
        if q > 0.0 {
            r.exercised += 1;
        } else {
            r.assigned += 1;
        }
        self.shares[i] = 0.0;
    }

    /// Cash moved by marking futures `i` to `price`.
    fn settle_future(&mut self, i: usize, price: f64, r: &mut StepReport) {
        let q = self.shares[i];
        if q != 0.0 && self.mark[i].is_finite() {
            let vm = q * self.instruments[i].multiplier * (price - self.mark[i]);
            self.cash += vm;
            r.variation_margin += vm;
        }
        self.mark[i] = price;
    }

    fn marks(&self, price: Vec<f64>, at_open_of: Option<usize>) -> Marks {
        let m = &self.market;
        let ppy = self.costs.periods_per_year;
        let t = at_open_of.unwrap_or(self.t);
        let need_vol = matches!(self.rules.margin, MarginModel::Portfolio(_));
        Marks::new(
            &self.instruments,
            price,
            &self.strike,
            &self.deliverable,
            self.rules.rate,
            need_vol,
            |i| {
                let e = self.instruments[i].expiry().unwrap_or(t) as f64;
                // At an open one more bar of life remains than at the close.
                let extra = if at_open_of.is_some() { 1.0 } else { 0.0 };
                (e - t as f64 + extra) / ppy
            },
            |i| match self.instruments[i].kind {
                Kind::Option(o) => m.sigma[m.at(t, o.underlying)] * ppy.sqrt(),
                _ => 0.0,
            },
        )
    }

    fn requirement(&self, units: &[f64], marks: &Marks, initial: bool) -> f64 {
        requirement(
            &self.rules.margin,
            initial,
            &self.instruments,
            &self.groups,
            units,
            marks,
        )
    }

    /// Largest `f` in `[0, 1]` with `requirement(book(f)) <= bound`, given
    /// that `book(0)` satisfies it.
    fn largest_feasible(&self, marks: &Marks, bound: f64, book: impl Fn(f64) -> Vec<f64>) -> f64 {
        if self.requirement(&book(1.0), marks, true) <= bound {
            return 1.0;
        }
        let (mut lo, mut hi) = (0.0, 1.0);
        for _ in 0..30 {
            let mid = 0.5 * (lo + hi);
            if self.requirement(&book(mid), marks, true) <= bound {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    fn step_with(&mut self, target: &[f64], mode: Target) -> Result<StepReport> {
        if self.done() {
            return Err(Error::Schema("simulation is finished".into()));
        }
        let m = Arc::clone(&self.market);
        let s = m.securities;
        if target.len() != s {
            return Err(Error::Dimension {
                context: "target".into(),
                expected: s,
                got: target.len(),
            });
        }
        if let Some(i) = target.iter().position(|x| x.is_infinite()) {
            return Err(Error::InvalidValue {
                context: "target".into(),
                index: i,
                value: target[i] as f32,
                reason: "infinite (use NaN to hold)",
            });
        }
        let (t0, t1) = (self.t, self.t + 1);
        let ppy = self.costs.periods_per_year;
        let mut r = StepReport {
            period: t1,
            margin_scale: 1.0,
            ..StepReport::default()
        };

        // 0. Early exercise decided at the close of t0 (before the ex-date
        //    of t1), and AM settlement of contracts whose last trading bar
        //    was t0.
        let thr = self.rules.exercise_threshold;
        for k in 0..self.options.len() {
            let i = self.options[k];
            let Kind::Option(o) = self.instruments[i].kind else {
                unreachable!()
            };
            if self.shares[i] == 0.0 {
                continue;
            }
            let u = o.underlying;
            if o.expiry == t0 {
                if let Settlement::Cash(SettlementPrice::NextOpen) = o.settlement {
                    let spot = settlement_open(&m, t0, t1, u);
                    if self.intrinsic(i, spot) >= thr * self.ratio(i) {
                        self.exercise(i, spot, &mut r);
                    } else {
                        self.shares[i] = 0.0;
                        r.expired += 1;
                    }
                }
                continue;
            }
            if !(self.rules.early_exercise
                && o.style == Style::American
                && o.expiry > t0
                && m.listed[m.at(t0, i)]
                && m.listed[m.at(t0, u)])
            {
                continue;
            }
            let spot = m.close[m.at(t0, u)];
            let intrinsic = self.intrinsic(i, spot);
            if intrinsic <= 0.0 {
                continue;
            }
            let time_value = m.close[m.at(t0, i)] - intrinsic;
            let go = match o.right {
                Right::Call => {
                    let d = if self.instruments[u].kind == Kind::Equity {
                        m.dividend[m.at(t1, u)] * self.ratio(i)
                    } else {
                        0.0
                    };
                    d > 0.0 && d > time_value
                }
                Right::Put => time_value <= thr * self.ratio(i),
            };
            if go {
                self.exercise(i, spot, &mut r);
            }
        }
        for k in 0..self.futures.len() {
            let i = self.futures[k];
            let Kind::Future(f) = self.instruments[i].kind else {
                unreachable!()
            };
            if f.expiry == t0 && self.shares[i] != 0.0 {
                if f.settlement == SettlementPrice::NextOpen {
                    let u = f
                        .underlying
                        .expect("validated: NextOpen futures have an underlying");
                    let px = settlement_open(&m, t0, t1, u);
                    self.settle_future(i, px, &mut r);
                }
                self.shares[i] = 0.0;
                r.expired += 1;
            }
        }

        // 1. Corporate actions at the open of t1.
        for i in 0..s {
            let q = self.shares[i];
            let inst = self.instruments[i];
            if let Kind::Option(o) = inst.kind {
                let (k, su) = (m.split[m.at(t1, i)], m.split[m.at(t1, o.underlying)]);
                if k != 1.0 || su != 1.0 {
                    let spec = instrument::OptionSpec {
                        strike: self.strike[i],
                        deliverable: self.deliverable[i],
                        ..o
                    };
                    (self.strike[i], self.deliverable[i]) = spec.adjust(su, k);
                    self.shares[i] = q * k;
                }
                continue;
            }
            if q != 0.0 && inst.kind == Kind::Equity {
                let d = m.dividend[m.at(t1, i)];
                if d != 0.0 {
                    self.cash += q * d;
                    r.dividends += q * d;
                }
                self.shares[i] = q * m.split[m.at(t1, i)];
            }
        }
        // 2. Carry: borrow on short stock (valued at the last close), cash/margin.
        for i in 0..s {
            if self.shares[i] < 0.0
                && self.instruments[i].kind == Kind::Equity
                && m.listed[m.at(t0, i)]
            {
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
                if self.instruments[i].is_future() {
                    self.settle_future(i, px, &mut r);
                } else {
                    self.cash += self.shares[i] * px * self.instruments[i].multiplier;
                }
                self.shares[i] = 0.0;
                r.delisted += 1;
            }
        }

        // 4. Opening prices; futures settle to the open; opening equity.
        let open = |i: usize| {
            if m.tradable[m.at(t1, i)] {
                m.open[m.at(t1, i)]
            } else {
                m.close[m.at(t0, i)] / m.split[m.at(t1, i)]
            }
        };
        for k in 0..self.futures.len() {
            let i = self.futures[k];
            if m.listed[m.at(t1, i)] {
                self.settle_future(i, open(i), &mut r);
            }
        }
        let value = |sim: &Self, i: usize, px: f64| -> f64 {
            if sim.instruments[i].is_future() {
                0.0 // settled
            } else {
                sim.shares[i] * px * sim.instruments[i].multiplier
            }
        };
        r.equity_open = self.cash
            + (0..s)
                .filter(|&i| self.shares[i] != 0.0)
                .map(|i| value(self, i, open(i)))
                .sum::<f64>();
        if r.equity_open.is_nan() || r.equity_open <= 0.0 {
            return Ok(self.ruin(r));
        }
        let e_open = r.equity_open;
        let tradable = |sim: &Self, i: usize| m.tradable[m.at(t1, i)] && !sim.is_index(i);
        let with_margin = !self.rules.margin.is_none();
        let open_marks = with_margin.then(|| self.marks((0..s).map(open).collect(), Some(t1)));

        // 5. A margin call from the last close: cut every tradable position
        //    by the same fraction until the book meets the requirement.
        let mut target: Vec<f64> = target.to_vec();
        let mut mode = mode;
        if self.margin_call {
            self.margin_call = false;
            let marks = open_marks.as_ref().expect("margin model");
            if self.requirement(&self.shares, marks, false) > e_open {
                let book = |f: f64| -> Vec<f64> {
                    (0..s)
                        .map(|i| {
                            if tradable(self, i) {
                                f * self.shares[i]
                            } else {
                                self.shares[i]
                            }
                        })
                        .collect()
                };
                let mut lo = 0.0;
                let mut hi = 1.0;
                for _ in 0..30 {
                    let mid = 0.5 * (lo + hi);
                    if self.requirement(&book(mid), marks, false) <= e_open {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                target = (0..s)
                    .map(|i| {
                        if tradable(self, i) && self.shares[i] != 0.0 {
                            r.liquidated += 1;
                            lo * self.shares[i]
                        } else {
                            f64::NAN
                        }
                    })
                    .collect();
                mode = Target::Units;
            }
        }
        let liquidating = r.liquidated > 0;

        // 6. Targets as weights, projected onto the constraints.
        let px_unit = |sim: &Self, i: usize| sim.unit_exposure(i, open);
        // A derivative whose underlying has no price cannot be sized by weight.
        let sizable = |sim: &Self, i: usize| {
            let p = px_unit(sim, i);
            tradable(sim, i) && p.is_finite() && p > 0.0
        };
        let units_in = mode == Target::Units;
        let w_in: Vec<f64> = if units_in {
            (0..s)
                .map(|i| target[i] * px_unit(self, i) / e_open)
                .collect()
        } else {
            target.clone()
        };
        let fixed_gross: f64 = (0..s)
            .filter(|&i| self.shares[i] != 0.0 && (!tradable(self, i) || w_in[i].is_nan()))
            .map(|i| (self.shares[i] * px_unit(self, i)).abs())
            .sum::<f64>()
            / e_open;
        let (w, projection) = if liquidating {
            // A liquidation only reduces positions; constraints cannot bind.
            (w_in.clone(), 0.0)
        } else {
            self.project(&w_in, |i| sizable(self, i), fixed_gross)
        };
        r.projection = projection;
        let mut want: Vec<f64> = (0..s)
            .map(|i| {
                if w[i].is_nan() {
                    self.shares[i]
                } else if units_in && w[i] == w_in[i] {
                    target[i] // exact: no constraint changed it
                } else {
                    w[i] * e_open / px_unit(self, i)
                }
            })
            .collect();
        for i in 0..s {
            let lot = self.instruments[i].lot;
            if lot > 0.0 && !w[i].is_nan() {
                // A liquidation rounds toward zero so it never keeps more
                // than it must.
                let n = want[i] / lot;
                want[i] = if liquidating { n.trunc() } else { n.round() } * lot;
            }
        }

        // 7. Initial margin: scale the trades back until the book meets the
        //    initial requirement, or does not raise an existing deficiency.
        if with_margin && !liquidating {
            let marks = open_marks.as_ref().expect("margin model");
            let now = self.requirement(&self.shares, marks, true);
            let bound = e_open.max(now);
            let book = |f: f64| -> Vec<f64> {
                (0..s)
                    .map(|i| self.shares[i] + f * (want[i] - self.shares[i]))
                    .collect()
            };
            let f = self.largest_feasible(marks, bound, book);
            if f < 1.0 {
                r.margin_scale = f;
                for (i, x) in want.iter_mut().enumerate() {
                    let (cur, lot) = (self.shares[i], self.instruments[i].lot);
                    let mut d = f * (*x - cur);
                    if lot > 0.0 {
                        // Round toward the current position, never past `f`.
                        d = (d / lot).trunc() * lot;
                    }
                    *x = cur + d;
                }
            }
        }

        // 8. Participation cap and fills.
        let bps = 1e-4;
        for i in 0..s {
            if w[i].is_nan() {
                continue; // held or untradable
            }
            let inst = self.instruments[i];
            let px = open(i);
            let mut q = want[i] - self.shares[i];
            if q == 0.0 {
                continue;
            }
            let adv = m.adv_shares[m.at(t1, i)];
            let mut cap = self.costs.max_participation * adv;
            if inst.lot > 0.0 {
                cap = (cap / inst.lot).floor() * inst.lot;
            }
            if q.abs() > cap {
                q = q.signum() * cap;
                r.clipped += 1;
            }
            if q == 0.0 {
                continue;
            }
            let notional = q.abs() * px * inst.multiplier;
            let impact_frac = if adv > 0.0 {
                self.costs.impact_coef * m.sigma[m.at(t1, i)] * (q.abs() / adv).sqrt()
            } else {
                0.0
            };
            let given = m.half_spread.as_ref().map_or(f64::NAN, |h| h[m.at(t1, i)]);
            let half = if !given.is_nan() {
                given
            } else if inst.is_option() {
                (px * self.costs.option_half_spread_bps * bps)
                    .max(self.costs.option_min_half_spread)
            } else {
                px * self.costs.half_spread_bps * bps
            };
            let spread = q.abs() * inst.multiplier * half;
            let impact = notional * impact_frac;
            let commission = match inst.kind {
                Kind::Option(_) => q.abs() * self.costs.option_commission,
                Kind::Future(_) => q.abs() * self.costs.future_commission,
                _ => notional * self.costs.commission_bps * bps,
            };
            let premium = if inst.is_future() {
                0.0
            } else {
                q * px * inst.multiplier
            };
            self.cash -= premium + spread + impact + commission;
            self.shares[i] += q;
            r.turnover += notional;
            r.spread_cost += spread;
            r.impact_cost += impact;
            r.commission += commission;
        }
        r.turnover /= e_open;

        // 9. Close of t1: futures settle; contracts whose last trading bar is
        //    t1 are settled (PM cash, physical) or expire.
        for k in 0..self.futures.len() {
            let i = self.futures[k];
            if m.listed[m.at(t1, i)] {
                self.settle_future(i, m.close[m.at(t1, i)], &mut r);
                let Kind::Future(f) = self.instruments[i].kind else {
                    unreachable!()
                };
                if f.expiry == t1 && f.settlement == SettlementPrice::Close && self.shares[i] != 0.0
                {
                    self.shares[i] = 0.0;
                    r.expired += 1;
                }
            }
        }
        for k in 0..self.options.len() {
            let i = self.options[k];
            let Kind::Option(o) = self.instruments[i].kind else {
                unreachable!()
            };
            if o.expiry != t1 || self.shares[i] == 0.0 {
                continue;
            }
            if o.settlement == Settlement::Cash(SettlementPrice::NextOpen) {
                continue; // settles at the next open
            }
            let spot = m.close[m.at(t1, o.underlying)];
            if self.intrinsic(i, spot) >= thr * self.ratio(i) {
                self.exercise(i, spot, &mut r);
            } else {
                self.shares[i] = 0.0;
                r.expired += 1;
            }
        }

        // 10. Mark to the close of t1.
        let mut long = 0.0;
        let mut short = 0.0;
        let mut held = 0.0;
        for i in 0..s {
            if self.shares[i] != 0.0 {
                let px = m.close[m.at(t1, i)];
                let x = self.shares[i] * self.unit_exposure(i, |j| m.close[m.at(t1, j)]);
                if x > 0.0 {
                    long += x
                } else {
                    short += x
                }
                held += value(self, i, px);
            }
        }
        let equity = self.cash + held;
        self.t = t1;
        if equity.is_nan() || equity <= 0.0 {
            return Ok(self.ruin(r));
        }
        r.equity = equity;
        r.log_return = (equity / self.equity).ln();
        r.gross = (long - short) / equity;
        r.net = (long + short) / equity;
        self.equity = equity;

        // 11. Maintenance margin at the close.
        if with_margin {
            let marks = self.marks((0..s).map(|i| m.close[m.at(t1, i)]).collect(), None);
            r.margin_requirement = self.requirement(&self.shares, &marks, false);
            if equity < r.margin_requirement {
                r.margin_call = true;
                self.margin_call = true;
            }
        }
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

/// The underlying's opening level on `t1` for an AM settlement, or its last
/// close if it did not open.
fn settlement_open(m: &Market, t0: usize, t1: usize, u: usize) -> f64 {
    let k = m.at(t1, u);
    let o = m.open[k];
    let index = matches!(m.instrument(u).kind, Kind::Index { .. });
    if (m.tradable[k] || (index && m.listed[k])) && o > 0.0 && o.is_finite() {
        o
    } else {
        m.close[m.at(t0, u)] / m.split[k]
    }
}

#[cfg(test)]
mod tests {
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
            instruments: Vec::new(),
            half_spread: None,
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

#[cfg(test)]
mod derivative_tests;
