//! Margin requirements.
//!
//! * No margin ([`MarginModel::None`]): no requirement; leverage is limited only by the
//!   gross-exposure constraint (the behavior before margin existed).
//! * Regulation T ([`MarginModel::RegT`]): strategy-based (Regulation T / FINRA 4210)
//!   requirements position by position: a fraction of market value for
//!   stock, full premium for long options (75% beyond nine months), the
//!   naked-writer formula for short options, and the exchange's fraction of
//!   notional for futures. Positions are not paired into spreads or covered
//!   writes, so hedged option books are charged as if unhedged; use
//!   [`MarginModel::Portfolio`] for those.
//! * Portfolio margin ([`MarginModel::Portfolio`]): risk-based (FINRA 4210(g) portfolio margin,
//!   OCC TIMS style). Positions are grouped by their root underlying and
//!   revalued over a grid of underlying moves (±15% for stocks, −8%/+6% for
//!   broad-based indices), optionally crossed with volatility shifts;
//!   options are revalued with [`pricing`](super::pricing) at volatilities
//!   implied from their own prices. A group's requirement is its worst
//!   loss, and at least a fixed amount per option contract. Hedges and
//!   spreads offset within a group, never across groups.
//!
//! Every requirement is positively homogeneous in the positions: scaling
//! all of them by `f >= 0` scales the requirement by `f`. It is not linear
//! across positions (hedges offset), so the simulator sizes margin-call
//! liquidations and initial-margin cutbacks by bisection on the fraction.

use super::instrument::{Instrument, Kind, Settlement, Style};
use super::pricing::{Contract, Right, VOL_MAX, VOL_MIN};
use crate::error::{Error, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct RegT {
    pub initial_long: f64,
    pub maintenance_long: f64,
    pub initial_short: f64,
    pub maintenance_short: f64,
    /// Short option: fraction of the underlying's value (20%; 15% for a
    /// broad-based index).
    pub option_base: f64,
    pub option_base_index: f64,
    /// Short option minimum: fraction of the underlying's value (calls) or
    /// of the aggregate exercise price (puts).
    pub option_minimum: f64,
}

impl Default for RegT {
    fn default() -> Self {
        RegT {
            initial_long: 0.5,
            maintenance_long: 0.25,
            initial_short: 0.5,
            maintenance_short: 0.30,
            option_base: 0.20,
            option_base_index: 0.15,
            option_minimum: 0.10,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Portfolio {
    /// Symmetric move range for stocks and narrow indices.
    pub equity_range: f64,
    pub index_down: f64,
    pub index_up: f64,
    /// Grid intervals across each range (`points + 1` scenarios).
    pub points: usize,
    /// Relative volatility shifts applied in addition to 0 (e.g. 0.25:
    /// every price move is also evaluated at 0.75x and 1.25x volatility).
    pub vol_shift: f64,
    /// Minimum per option contract, per unit of multiplier (0.375: $37.50
    /// for a 100-multiplier contract).
    pub min_per_contract: f64,
}

impl Default for Portfolio {
    fn default() -> Self {
        Portfolio {
            equity_range: 0.15,
            index_down: 0.08,
            index_up: 0.06,
            points: 10,
            vol_shift: 0.0,
            min_per_contract: 0.375,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub enum MarginModel {
    #[default]
    None,
    RegT(RegT),
    Portfolio(Portfolio),
}

impl MarginModel {
    pub(super) fn validate(&self) -> Result<()> {
        let unit = |x: f64| (0.0..=1.0).contains(&x);
        let ok = match self {
            MarginModel::None => true,
            MarginModel::RegT(r) => {
                [
                    r.initial_long,
                    r.maintenance_long,
                    r.initial_short,
                    r.maintenance_short,
                    r.option_base,
                    r.option_base_index,
                    r.option_minimum,
                ]
                .into_iter()
                .all(unit)
                    && r.maintenance_long <= r.initial_long
                    && r.maintenance_short <= r.initial_short
            }
            MarginModel::Portfolio(p) => {
                unit(p.equity_range)
                    && unit(p.index_down)
                    && unit(p.index_up)
                    && p.points >= 1
                    && p.points <= 1000
                    && (0.0..1.0).contains(&p.vol_shift)
                    && p.min_per_contract >= 0.0
                    && p.min_per_contract.is_finite()
            }
        };
        if ok {
            Ok(())
        } else {
            Err(Error::Schema(format!(
                "invalid margin parameters: {self:?}"
            )))
        }
    }

    pub fn is_none(&self) -> bool {
        matches!(self, MarginModel::None)
    }
}

/// Root underlying of every instrument and the portfolio-margin groups.
#[derive(Clone, Debug)]
pub(super) struct Groups {
    /// Instruments by root underlying (empty for non-roots).
    members: Vec<Vec<usize>>,
}

impl Groups {
    pub fn new(instruments: &[Instrument]) -> Self {
        fn root(all: &[Instrument], i: usize) -> usize {
            match all[i].kind {
                Kind::Equity | Kind::Index { .. } => i,
                Kind::Option(o) => root(all, o.underlying),
                Kind::Future(f) => f.underlying.unwrap_or(i),
            }
        }
        let mut members = vec![Vec::new(); instruments.len()];
        for i in 0..instruments.len() {
            members[root(instruments, i)].push(i);
        }
        Groups { members }
    }
}

/// Prices and option model inputs at one evaluation point (an open or a
/// close), computed once and reused for any number of candidate books.
#[derive(Clone, Debug)]
pub(super) struct Marks {
    pub price: Vec<f64>,
    /// Per option: years to expiry, volatility, strike, deliverable.
    pub tau: Vec<f64>,
    pub vol: Vec<f64>,
    pub strike: Vec<f64>,
    pub deliverable: Vec<f64>,
    /// Per option: underlying value delivered per unit of quoted price
    /// basis, `deliverable * underlying multiplier / multiplier` (1 for a
    /// standard or cash-settled contract). The model prices one delivered
    /// unit at strike `strike / ratio`; a contract is worth
    /// `ratio * multiplier` of those.
    pub ratio: Vec<f64>,
    pub rate: f64,
}

impl Marks {
    /// `tau(i)`: years to expiry of option `i`; `fallback_vol(i)`:
    /// annualized volatility to use when option `i`'s price implies none
    /// (stale or off-model prices).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        instruments: &[Instrument],
        price: Vec<f64>,
        strike: &[f64],
        deliverable: &[f64],
        rate: f64,
        need_vol: bool,
        tau: impl Fn(usize) -> f64,
        fallback_vol: impl Fn(usize) -> f64,
    ) -> Self {
        let n = instruments.len();
        let mut m = Marks {
            price,
            tau: vec![0.0; n],
            vol: vec![0.0; n],
            strike: strike.to_vec(),
            deliverable: deliverable.to_vec(),
            ratio: vec![1.0; n],
            rate,
        };
        for (i, inst) in instruments.iter().enumerate() {
            if let Kind::Option(o) = inst.kind {
                m.tau[i] = tau(i).max(0.0);
                if o.settlement == Settlement::Physical {
                    m.ratio[i] =
                        m.deliverable[i] * instruments[o.underlying].multiplier / inst.multiplier;
                }
                if need_vol {
                    let c = m.contract(instruments, i, o.underlying, o.right, 0.0);
                    m.vol[i] = c
                        .implied_vol(m.price[i] / m.ratio[i])
                        .unwrap_or_else(|| fallback_vol(i))
                        .clamp(VOL_MIN, VOL_MAX);
                }
            }
        }
        m
    }

    fn contract(&self, all: &[Instrument], i: usize, u: usize, right: Right, mv: f64) -> Contract {
        let carry = if all[u].is_future() { 0.0 } else { self.rate };
        Contract {
            right,
            spot: self.price[u] * (1.0 + mv),
            strike: self.strike[i] / self.ratio[i],
            tau: self.tau[i],
            rate: self.rate,
            carry,
        }
    }
}

/// Requirement of the book `units` under `model` at `marks`.
pub(super) fn requirement(
    model: &MarginModel,
    initial: bool,
    instruments: &[Instrument],
    groups: &Groups,
    units: &[f64],
    marks: &Marks,
) -> f64 {
    match model {
        MarginModel::None => 0.0,
        MarginModel::RegT(r) => reg_t(r, initial, instruments, units, marks),
        MarginModel::Portfolio(p) => portfolio(p, instruments, groups, units, marks),
    }
}

fn reg_t(r: &RegT, initial: bool, all: &[Instrument], units: &[f64], m: &Marks) -> f64 {
    let mut req = 0.0;
    for (i, &q) in units.iter().enumerate() {
        if q == 0.0 {
            continue;
        }
        let inst = &all[i];
        let value = q.abs() * m.price[i] * inst.multiplier;
        req += match inst.kind {
            Kind::Equity => {
                let f = match (q > 0.0, initial) {
                    (true, true) => r.initial_long,
                    (true, false) => r.maintenance_long,
                    (false, true) => r.initial_short,
                    (false, false) => r.maintenance_short,
                };
                f * value
            }
            Kind::Index { .. } => 0.0,
            Kind::Future(f) => {
                value
                    * if initial {
                        f.initial_margin
                    } else {
                        f.maintenance_margin
                    }
            }
            Kind::Option(_) if q > 0.0 => value * if m.tau[i] > 0.75 { 0.75 } else { 1.0 },
            Kind::Option(o) => {
                let u = &all[o.underlying];
                let uv = m.price[o.underlying] * u.multiplier * m.deliverable[i];
                let kv = m.strike[i] * inst.multiplier;
                let base = match u.kind {
                    Kind::Index { broad_based: true } => r.option_base_index,
                    _ => r.option_base,
                };
                let (otm, floor) = match o.right {
                    Right::Call => ((kv - uv).max(0.0), r.option_minimum * uv),
                    Right::Put => ((uv - kv).max(0.0), r.option_minimum * kv),
                };
                value + q.abs() * (base * uv - otm).max(floor)
            }
        };
    }
    req
}

fn portfolio(p: &Portfolio, all: &[Instrument], g: &Groups, units: &[f64], m: &Marks) -> f64 {
    let shifts: &[f64] = if p.vol_shift > 0.0 {
        &[-1.0, 0.0, 1.0]
    } else {
        &[0.0]
    };
    let mut total = 0.0;
    let mut pnl = vec![0.0; (p.points + 1) * shifts.len()];
    for (root, members) in g.members.iter().enumerate() {
        if !members.iter().any(|&i| units[i] != 0.0) {
            continue;
        }
        let (down, up) = match all[root].kind {
            Kind::Index { broad_based: true } => (p.index_down, p.index_up),
            Kind::Future(f) => (f.maintenance_margin, f.maintenance_margin),
            _ => (p.equity_range, p.equity_range),
        };
        pnl.iter_mut().for_each(|x| *x = 0.0);
        let mut minimum = 0.0;
        for &i in members {
            let q = units[i];
            if q == 0.0 {
                continue;
            }
            let inst = &all[i];
            let scale = q * inst.multiplier;
            match inst.kind {
                Kind::Index { .. } => {}
                Kind::Equity | Kind::Future(_) => {
                    let v = scale * m.price[i];
                    for j in 0..=p.points {
                        let mv = -down + (down + up) * j as f64 / p.points as f64;
                        for k in 0..shifts.len() {
                            pnl[j * shifts.len() + k] += v * mv;
                        }
                    }
                }
                Kind::Option(o) => {
                    minimum += q.abs() * p.min_per_contract * inst.multiplier;
                    let american = o.style == Style::American;
                    let per = m.ratio[i];
                    let base = m.contract(all, i, o.underlying, o.right, 0.0);
                    let v0 = base.value(m.vol[i], american);
                    for j in 0..=p.points {
                        let mv = -down + (down + up) * j as f64 / p.points as f64;
                        let c = m.contract(all, i, o.underlying, o.right, mv);
                        for (k, &s) in shifts.iter().enumerate() {
                            let vol = (m.vol[i] * (1.0 + s * p.vol_shift)).clamp(VOL_MIN, VOL_MAX);
                            pnl[j * shifts.len() + k] +=
                                scale * per * (c.value(vol, american) - v0);
                        }
                    }
                }
            }
        }
        let worst = pnl.iter().copied().fold(f64::INFINITY, f64::min);
        total += (-worst).max(0.0).max(minimum);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::super::instrument::{OptionSpec, SettlementPrice};
    use super::*;

    fn book() -> Vec<Instrument> {
        let opt = |right, strike| Instrument {
            kind: Kind::Option(OptionSpec {
                underlying: 0,
                right,
                strike,
                expiry: 100,
                style: Style::American,
                settlement: Settlement::Physical,
                deliverable: 100.0,
            }),
            multiplier: 100.0,
            lot: 1.0,
        };
        vec![
            Instrument::EQUITY,
            opt(Right::Call, 110.0),
            opt(Right::Put, 90.0),
        ]
    }

    fn marks(all: &[Instrument], prices: Vec<f64>) -> Marks {
        let strike: Vec<f64> = all
            .iter()
            .map(|x| match x.kind {
                Kind::Option(o) => o.strike,
                _ => 0.0,
            })
            .collect();
        Marks::new(
            all,
            prices,
            &strike,
            &[0.0, 100.0, 100.0],
            0.0,
            true,
            |_| 0.25,
            |_| 0.3,
        )
    }

    #[test]
    fn reg_t_matches_the_rulebook() {
        let all = book();
        let m = marks(&all, vec![100.0, 2.0, 1.5]);
        let g = Groups::new(&all);
        let r = MarginModel::RegT(RegT::default());
        // 100 shares long: 50% initial, 25% maintenance.
        assert_eq!(
            requirement(&r, true, &all, &g, &[100.0, 0.0, 0.0], &m),
            5000.0
        );
        assert_eq!(
            requirement(&r, false, &all, &g, &[100.0, 0.0, 0.0], &m),
            2500.0
        );
        // One naked call, 10 OTM: 200 + max(2000 - 1000, 1000) = 1200.
        assert_eq!(
            requirement(&r, false, &all, &g, &[0.0, -1.0, 0.0], &m),
            1200.0
        );
        // One naked put, 10 OTM: 150 + max(2000 - 1000, 900) = 1150.
        assert_eq!(
            requirement(&r, false, &all, &g, &[0.0, 0.0, -1.0], &m),
            1150.0
        );
        // A long option needs its full premium.
        assert_eq!(
            requirement(&r, false, &all, &g, &[0.0, 2.0, 0.0], &m),
            400.0
        );
    }

    #[test]
    fn portfolio_offsets_hedges_within_a_group() {
        let all = book();
        let m = marks(&all, vec![100.0, 2.0, 1.5]);
        let g = Groups::new(&all);
        let p = MarginModel::Portfolio(Portfolio::default());
        let stock = requirement(&p, false, &all, &g, &[100.0, 0.0, 0.0], &m);
        assert!((stock - 1500.0).abs() < 1e-9); // 15% down
        // A protective put caps the downside: less than the stock alone.
        let hedged = requirement(&p, false, &all, &g, &[100.0, 0.0, 1.0], &m);
        assert!(hedged < stock && hedged > 0.0, "{hedged}");
        // Homogeneous of degree one.
        let twice = requirement(&p, false, &all, &g, &[200.0, 0.0, 2.0], &m);
        assert!((twice - 2.0 * hedged).abs() < 1e-6);
        // The per-contract minimum applies to a far out-of-the-money option.
        let far = requirement(&p, false, &all, &g, &[0.0, 1.0, 0.0], &m);
        assert!(far >= 37.5);
    }

    #[test]
    fn groups_follow_root_underlyings() {
        let mut all = book();
        all.push(Instrument {
            kind: Kind::Future(super::super::instrument::FutureSpec {
                underlying: Some(0),
                expiry: 50,
                settlement: SettlementPrice::Close,
                initial_margin: 0.1,
                maintenance_margin: 0.08,
            }),
            multiplier: 10.0,
            lot: 1.0,
        });
        let g = Groups::new(&all);
        assert_eq!(g.members[0], vec![0, 1, 2, 3]);
        assert!(g.members[1..].iter().all(Vec::is_empty));
    }
}
