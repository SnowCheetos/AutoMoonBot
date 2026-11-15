//! What each column of a [`Market`](super::Market) is: a stock, a
//! non-tradable index level, an option, or a future.
//!
//! Every price is quoted per unit of the underlying (a share, an index
//! point, a point of the future), and a position is a count of *units*:
//! shares, or contracts. One contract is worth `multiplier × price`.
//!
//! Options carry their exercise terms as a *deliverable* (underlying units
//! per contract) and a strike, with an aggregate exercise price of
//! `strike × multiplier` per contract, as the OCC defines them. Both are
//! adjusted when the underlying or the option splits (see
//! [`OptionSpec::adjust`]), so a standard 2-for-1 split and a non-standard
//! adjustment that changes the deliverable are both exact.

use super::pricing::Right;
use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Style {
    American,
    European,
}

/// The price a cash settlement uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SettlementPrice {
    /// The underlying's close on the last trading bar (PM settlement).
    Close,
    /// The underlying's open on the bar after the last trading bar (AM
    /// settlement, e.g. monthly SPX options and equity index futures).
    NextOpen,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Settlement {
    /// Exercise delivers (or takes) the deliverable at the strike.
    Physical,
    Cash(SettlementPrice),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptionSpec {
    /// Column of the underlying: a stock, an index or a future.
    pub underlying: usize,
    pub right: Right,
    /// Strike per unit of the underlying.
    pub strike: f64,
    /// Last trading bar. Exercise happens after its close; the option must
    /// not be listed after it.
    pub expiry: usize,
    pub style: Style,
    pub settlement: Settlement,
    /// Underlying units delivered per contract (normally the multiplier).
    pub deliverable: f64,
}

impl OptionSpec {
    /// Terms after the underlying splits by `underlying_split` (new units
    /// per old) and the option's own contract count is scaled by
    /// `contract_split` on the same bar. The aggregate exercise price and
    /// the delivered quantity of the underlying are preserved per original
    /// contract:
    ///
    /// * a standard 2-for-1 (both ratios 2): twice the contracts, half the
    ///   strike, same deliverable per contract;
    /// * a 3-for-2 split adjusted OCC-style (underlying 1.5, contracts 1):
    ///   same contracts and strike, deliverable 150 shares.
    pub fn adjust(&self, underlying_split: f64, contract_split: f64) -> (f64, f64) {
        (
            self.strike / contract_split,
            self.deliverable * underlying_split / contract_split,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FutureSpec {
    /// The index or stock the future tracks, if it is in the market. Its
    /// margin scenarios move with that underlying's.
    pub underlying: Option<usize>,
    /// Last trading bar; the position is cash-settled at `settlement`.
    pub expiry: usize,
    pub settlement: SettlementPrice,
    /// Exchange margin per contract as a fraction of notional.
    pub initial_margin: f64,
    pub maintenance_margin: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Equity,
    /// A reference level (e.g. an index) that options and futures settle
    /// on. It is never traded or held.
    Index {
        broad_based: bool,
    },
    Option(OptionSpec),
    Future(FutureSpec),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Instrument {
    pub kind: Kind,
    /// Currency per unit of price per unit held (1 for shares, 100 for a
    /// standard equity option, 50 for an E-mini S&P future).
    pub multiplier: f64,
    /// Smallest position increment (0: any size). Trades round the target
    /// position to a whole number of lots.
    pub lot: f64,
}

impl Instrument {
    pub const EQUITY: Instrument = Instrument {
        kind: Kind::Equity,
        multiplier: 1.0,
        lot: 0.0,
    };

    pub fn is_option(&self) -> bool {
        matches!(self.kind, Kind::Option(_))
    }

    pub fn is_future(&self) -> bool {
        matches!(self.kind, Kind::Future(_))
    }

    pub fn expiry(&self) -> Option<usize> {
        match self.kind {
            Kind::Option(o) => Some(o.expiry),
            Kind::Future(f) => Some(f.expiry),
            _ => None,
        }
    }

    pub(super) fn validate(&self, i: usize, all: &[Instrument]) -> Result<()> {
        let bad = |what: String| Err(Error::Schema(format!("instrument {i}: {what}")));
        let pos = |x: f64| x > 0.0 && x.is_finite();
        if !pos(self.multiplier) {
            return bad("multiplier must be positive and finite".into());
        }
        if !(self.lot >= 0.0 && self.lot.is_finite()) {
            return bad("lot must be >= 0 and finite".into());
        }
        match self.kind {
            Kind::Equity | Kind::Index { .. } => {}
            Kind::Option(o) => {
                let Some(u) = all.get(o.underlying) else {
                    return bad(format!("underlying {} out of range", o.underlying));
                };
                if o.underlying == i || u.is_option() {
                    return bad("an option's underlying must be a stock, index or future".into());
                }
                if !pos(o.strike) || !pos(o.deliverable) {
                    return bad("strike and deliverable must be positive".into());
                }
                if o.settlement == Settlement::Physical && matches!(u.kind, Kind::Index { .. }) {
                    return bad("an index cannot be delivered; use cash settlement".into());
                }
            }
            Kind::Future(f) => {
                if let Some(u) = f.underlying {
                    match all.get(u).map(|x| x.kind) {
                        Some(Kind::Equity | Kind::Index { .. }) if u != i => {}
                        _ => return bad("a future's underlying must be a stock or index".into()),
                    }
                }
                if f.settlement == SettlementPrice::NextOpen && f.underlying.is_none() {
                    return bad("an AM-settled future needs its underlying in the market".into());
                }
                if !(pos(f.maintenance_margin) && f.initial_margin >= f.maintenance_margin)
                    || !f.initial_margin.is_finite()
                {
                    return bad("need 0 < maintenance_margin <= initial_margin".into());
                }
            }
        }
        Ok(())
    }
}
