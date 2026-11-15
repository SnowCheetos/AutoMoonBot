//! Options, futures and margin in the simulator.

use std::sync::Arc;

use super::instrument::{
    FutureSpec, Instrument, Kind, OptionSpec, Settlement, SettlementPrice, Style,
};
use super::margin::{MarginModel, Portfolio, RegT};
use super::pricing::Right;
use super::*;

const CAP: f64 = 100_000.0;

/// `[T][S]` open/close rows; every column listed and tradable.
fn market(open: &[&[f64]], close: &[&[f64]], instruments: Vec<Instrument>) -> Market {
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
        instruments,
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
        option_commission: 0.0,
        future_commission: 0.0,
        option_half_spread_bps: 0.0,
        option_min_half_spread: 0.0,
        ..Costs::default()
    }
}

fn levered(gross: f64) -> Constraints {
    Constraints {
        long_only: false,
        max_gross: gross,
        max_weight: gross,
    }
}

fn option(
    underlying: usize,
    right: Right,
    strike: f64,
    expiry: usize,
    settlement: Settlement,
) -> Instrument {
    Instrument {
        kind: Kind::Option(OptionSpec {
            underlying,
            right,
            strike,
            expiry,
            style: Style::American,
            settlement,
            deliverable: 100.0,
        }),
        multiplier: 100.0,
        lot: 1.0,
    }
}

/// Columns after `expiry` are delisted.
fn expire(m: &mut Market, col: usize, expiry: usize) {
    for t in expiry + 1..m.periods {
        let k = t * m.securities + col;
        m.listed[k] = false;
        m.tradable[k] = false;
    }
}

fn sim(m: Market, c: Constraints, rules: Rules) -> Simulator {
    Simulator::new(Arc::new(m), free(), c, 0, CAP)
        .unwrap()
        .with_rules(rules)
        .unwrap()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6 * a.abs().max(b.abs()).max(1.0)
}

#[test]
fn long_call_is_exercised_into_stock_at_expiry() {
    // Stock 100 -> 120; a 110 call bought at 2 and held to expiry at t=2.
    let mut m = market(
        &[&[100.0, 2.0], &[100.0, 2.0], &[118.0, 8.0], &[120.0, 0.0]],
        &[&[100.0, 2.0], &[100.0, 2.0], &[120.0, 10.0], &[121.0, 0.0]],
        vec![
            Instrument::EQUITY,
            option(0, Right::Call, 110.0, 2, Settlement::Physical),
        ],
    );
    expire(&mut m, 1, 2);
    let mut s = sim(m, levered(1.0), Rules::default());
    let r = s.step_units(&[0.0, 3.0]).unwrap();
    assert_eq!(s.shares()[1], 3.0);
    assert!(close(r.equity, CAP), "{r:?}"); // bought at 2, marked at 2
    let r = s.step_units(&[0.0, f64::NAN]).unwrap();
    assert_eq!(r.exercised, 1);
    assert_eq!(s.shares(), &[300.0, 0.0]);
    // Paid 600 premium and 33,000 at the strike for stock worth 36,000.
    assert!(close(s.cash(), CAP - 600.0 - 33_000.0), "{}", s.cash());
    assert!(close(r.equity, CAP - 600.0 + 3_000.0), "{r:?}");
    let r = s.step_units(&[f64::NAN, f64::NAN]).unwrap();
    assert!(close(r.equity, CAP - 600.0 + 3_000.0 + 300.0), "{r:?}");
}

#[test]
fn short_put_assignment_and_worthless_expiry() {
    // Two puts on one stock: 105 (ends in the money) and 90 (ends out).
    let mut m = market(
        &[&[100.0, 4.0, 1.0], &[100.0, 4.0, 1.0], &[100.0, 5.0, 0.2]],
        &[&[100.0, 4.0, 1.0], &[100.0, 4.0, 1.0], &[100.0, 5.0, 0.01]],
        vec![
            Instrument::EQUITY,
            option(0, Right::Put, 105.0, 2, Settlement::Physical),
            option(0, Right::Put, 90.0, 2, Settlement::Physical),
        ],
    );
    expire(&mut m, 1, 2);
    expire(&mut m, 2, 2);
    let rules = Rules {
        early_exercise: false,
        ..Rules::default()
    };
    let mut s = sim(m, levered(2.0), rules);
    s.step_units(&[0.0, -2.0, -2.0]).unwrap();
    // Sold 2 x 400 + 2 x 100 of premium.
    assert!(close(s.cash(), CAP + 1_000.0));
    let r = s.step_units(&[f64::NAN, f64::NAN, f64::NAN]).unwrap();
    assert_eq!((r.assigned, r.expired), (1, 1));
    // Assigned: bought 200 shares at 105.
    assert_eq!(s.shares(), &[200.0, 0.0, 0.0]);
    assert!(close(s.cash(), CAP + 1_000.0 - 21_000.0));
    assert!(close(r.equity, CAP + 1_000.0 - 1_000.0), "{r:?}");
}

#[test]
fn index_option_settles_on_the_next_open() {
    // Column 0 is an index (never traded); the call is AM-settled.
    let mut m = market(
        &[
            &[4000.0, 50.0],
            &[4000.0, 50.0],
            &[4050.0, 80.0],
            &[4100.0, 0.0],
        ],
        &[
            &[4000.0, 50.0],
            &[4000.0, 50.0],
            &[4080.0, 90.0],
            &[4120.0, 0.0],
        ],
        vec![
            Instrument {
                kind: Kind::Index { broad_based: true },
                multiplier: 1.0,
                lot: 0.0,
            },
            option(
                0,
                Right::Call,
                4000.0,
                2,
                Settlement::Cash(SettlementPrice::NextOpen),
            ),
        ],
    );
    for t in 0..4 {
        m.tradable[t * 2] = false;
    }
    expire(&mut m, 1, 2);
    // One contract is 400,000 of index notional: 4x the capital.
    let mut s = sim(m, levered(5.0), Rules::default());
    // Asking to trade the index does nothing.
    let r = s.step_units(&[5.0, 1.0]).unwrap();
    assert_eq!(s.shares(), &[0.0, 1.0]);
    assert!(r.projection > 0.0);
    let r = s.step_units(&[f64::NAN, f64::NAN]).unwrap();
    assert_eq!(r.expired + r.exercised, 0); // still held over the close
    assert!(close(r.equity, CAP - 5_000.0 + 9_000.0));
    // Settled at the next open (4100): 100 x 100.
    let r = s.step_units(&[f64::NAN, f64::NAN]).unwrap();
    assert_eq!(r.exercised, 1);
    assert_eq!(s.shares()[1], 0.0);
    assert!(close(r.equity, CAP - 5_000.0 + 10_000.0), "{r:?}");
}

fn future(expiry: usize, settlement: SettlementPrice, underlying: Option<usize>) -> Instrument {
    Instrument {
        kind: Kind::Future(FutureSpec {
            underlying,
            expiry,
            settlement,
            initial_margin: 0.06,
            maintenance_margin: 0.05,
        }),
        multiplier: 50.0,
        lot: 1.0,
    }
}

#[test]
fn futures_mark_to_market_and_settle() {
    let mut m = market(
        &[&[5000.0], &[5010.0], &[4990.0], &[5000.0]],
        &[&[5000.0], &[5020.0], &[5030.0], &[5000.0]],
        vec![future(2, SettlementPrice::Close, None)],
    );
    expire(&mut m, 0, 2);
    let mut s = sim(m, levered(10.0), Rules::default());
    // Two contracts bought at the 5010 open: notional 501,000 (5x equity).
    let r = s.step_units(&[2.0]).unwrap();
    assert!(close(r.variation_margin, 2.0 * 50.0 * 10.0));
    assert!(close(r.equity, CAP + 1_000.0), "{r:?}");
    assert!(close(r.gross, 2.0 * 50.0 * 5020.0 / r.equity));
    // Held into the last trading bar, settled at its close (5030).
    let r = s.step_units(&[f64::NAN]).unwrap();
    assert_eq!(r.expired, 1);
    assert!(close(r.equity, CAP + 1_000.0 + 1_000.0), "{r:?}");
    assert!(close(s.cash(), r.equity));
    assert_eq!(s.shares()[0], 0.0);
}

#[test]
fn am_settled_future_uses_the_underlying_open() {
    let ix = Instrument {
        kind: Kind::Index { broad_based: true },
        multiplier: 1.0,
        lot: 0.0,
    };
    let mut m = market(
        &[&[100.0, 100.0], &[100.0, 100.0], &[100.0, 0.0]],
        &[&[100.0, 100.0], &[101.0, 101.0], &[106.0, 0.0]],
        vec![ix, future(1, SettlementPrice::NextOpen, Some(0))],
    );
    m.open[4] = 104.0; // index open at t=2
    for t in 0..3 {
        m.tradable[t * 2] = false;
    }
    expire(&mut m, 1, 1);
    let mut s = sim(m, levered(10.0), Rules::default());
    s.step_units(&[0.0, 1.0]).unwrap();
    let r = s.step_units(&[0.0, f64::NAN]).unwrap();
    assert_eq!(r.expired, 1);
    // 101 -> 104 settlement on one contract of 50.
    assert!(close(r.variation_margin, 150.0), "{r:?}");
    assert!(close(r.equity, CAP + 50.0 + 150.0));
}

#[test]
fn splits_adjust_option_terms() {
    // 2-for-1 split at t=2 (standard: contracts double), then a 3-for-2
    // split at t=3 adjusted OCC-style (deliverable 150 per contract).
    let mut m = market(
        &[
            &[100.0, 10.0],
            &[100.0, 10.0],
            &[50.0, 5.0],
            &[40.0, 5.0],
            &[40.0, 0.0],
        ],
        &[
            &[100.0, 10.0],
            &[100.0, 10.0],
            &[60.0, 5.0],
            &[45.0, 5.0],
            &[40.0, 0.0],
        ],
        vec![
            Instrument::EQUITY,
            option(0, Right::Call, 90.0, 3, Settlement::Physical),
        ],
    );
    m.split[2 * 2] = 2.0;
    m.split[2 * 2 + 1] = 2.0;
    m.split[3 * 2] = 1.5;
    expire(&mut m, 1, 3);
    let rules = Rules {
        early_exercise: false,
        ..Rules::default()
    };
    let mut s = sim(m, levered(1.0), rules);
    s.step_units(&[0.0, 1.0]).unwrap();
    s.step_units(&[0.0, f64::NAN]).unwrap();
    assert_eq!(s.shares()[1], 2.0);
    assert_eq!(s.option_terms(1), Some((45.0, 100.0)));
    let r = s.step_units(&[f64::NAN, f64::NAN]).unwrap();
    // After the 3-for-2: deliverable 150, strike 45 (aggregate 4,500).
    // At expiry the stock is 45 > 45 x 100 / 150 = 30 per delivered share.
    assert_eq!(r.exercised, 1);
    assert_eq!(s.shares()[0], 2.0 * 150.0);
    assert!(
        close(s.cash(), CAP - 1_000.0 - 2.0 * 4_500.0),
        "{}",
        s.cash()
    );
}

#[test]
fn early_assignment_before_an_ex_dividend_bar() {
    // Deep in-the-money call (strike 50, stock 100) worth 50.5 the day
    // before a 2.00 dividend: the holder exercises; the writer is assigned
    // and owes the dividend on the shares it is now short.
    let mut m = market(
        &[&[100.0, 50.5], &[100.0, 50.5], &[98.0, 48.0], &[98.0, 48.0]],
        &[&[100.0, 50.5], &[100.0, 50.5], &[98.0, 48.0], &[98.0, 48.0]],
        vec![
            Instrument::EQUITY,
            option(0, Right::Call, 50.0, 3, Settlement::Physical),
        ],
    );
    m.dividend[2 * 2] = 2.0;
    let mut s = sim(m, levered(3.0), Rules::default());
    s.step_units(&[0.0, -1.0]).unwrap();
    let r = s.step_units(&[f64::NAN, f64::NAN]).unwrap();
    assert_eq!(r.assigned, 1);
    assert_eq!(s.shares(), &[-100.0, 0.0]);
    assert!(close(r.dividends, -200.0));
    // Premium 5,050 plus 5,000 at the strike, less the dividend, against
    // 100 shares short at 98.
    assert!(
        close(r.equity, CAP + 5_050.0 + 5_000.0 - 200.0 - 9_800.0),
        "{r:?}"
    );
}

#[test]
fn early_put_exercise_at_parity() {
    let m = market(
        &[&[100.0, 50.0], &[100.0, 50.0], &[50.0, 50.0], &[50.0, 50.0]],
        &[&[100.0, 50.0], &[50.0, 50.0], &[50.0, 50.0], &[50.0, 50.0]],
        vec![
            Instrument::EQUITY,
            option(0, Right::Put, 100.0, 3, Settlement::Physical),
        ],
    );
    let mut s = sim(m, levered(3.0), Rules::default());
    s.step_units(&[0.0, 1.0]).unwrap();
    // The put closed at its intrinsic value (no time value): exercised.
    let r = s.step_units(&[f64::NAN, f64::NAN]).unwrap();
    assert_eq!(r.exercised, 1);
    assert_eq!(s.shares(), &[-100.0, 0.0]);
    let off = Rules {
        early_exercise: false,
        ..Rules::default()
    };
    let m = market(
        &[&[100.0, 50.0], &[100.0, 50.0], &[50.0, 50.0], &[50.0, 50.0]],
        &[&[100.0, 50.0], &[50.0, 50.0], &[50.0, 50.0], &[50.0, 50.0]],
        vec![
            Instrument::EQUITY,
            option(0, Right::Put, 100.0, 3, Settlement::Physical),
        ],
    );
    let mut s = sim(m, levered(3.0), off);
    s.step_units(&[0.0, 1.0]).unwrap();
    assert_eq!(s.step_units(&[f64::NAN, f64::NAN]).unwrap().exercised, 0);
}

#[test]
fn initial_margin_scales_requested_leverage() {
    let m = market(
        &[&[10.0], &[10.0], &[10.0]],
        &[&[10.0], &[10.0], &[10.0]],
        vec![],
    );
    let rules = Rules {
        margin: MarginModel::RegT(RegT::default()),
        ..Rules::default()
    };
    let mut s = sim(m, levered(5.0), rules);
    // 3x requested; Reg T's 50% initial allows 2x.
    let r = s.step(&[3.0]).unwrap();
    assert!((r.margin_scale - 2.0 / 3.0).abs() < 1e-6, "{r:?}");
    assert!((r.gross - 2.0).abs() < 1e-6);
    assert!(!r.margin_call);
    assert!(close(r.margin_requirement, 0.25 * 2.0 * CAP));
    // Without margin the gross constraint alone applies.
    let m = market(&[&[10.0], &[10.0]], &[&[10.0], &[10.0]], vec![]);
    let mut s = sim(m, levered(5.0), Rules::default());
    assert!((s.step(&[3.0]).unwrap().gross - 3.0).abs() < 1e-9);
}

#[test]
fn margin_call_liquidates_at_the_next_open() {
    // 2x long; the price falls 40% at the close of t=2: equity 200 against
    // a 25% maintenance requirement of 300 on 1,200 of stock.
    let m = market(
        &[&[10.0], &[10.0], &[10.0], &[6.0], &[6.0]],
        &[&[10.0], &[10.0], &[6.0], &[6.0], &[6.0]],
        vec![],
    );
    let rules = Rules {
        margin: MarginModel::RegT(RegT::default()),
        ..Rules::default()
    };
    let mut s = Simulator::new(Arc::new(m), free(), levered(2.0), 0, 1_000.0)
        .unwrap()
        .with_rules(rules)
        .unwrap();
    s.step(&[2.0]).unwrap();
    let r = s.step(&[f64::NAN]).unwrap();
    assert!(close(r.equity, 200.0) && r.margin_call, "{r:?}");
    // The strategy asks for more; the call cuts the position instead, to
    // the size whose maintenance requirement equity covers: 200 / 0.25.
    let r = s.step(&[2.0]).unwrap();
    assert_eq!(r.liquidated, 1);
    assert!(close(s.shares()[0] * 6.0, 800.0), "{}", s.shares()[0]);
    assert!(!r.margin_call);
    // The next bar trades normally again.
    let r = s.step(&[0.5]).unwrap();
    assert_eq!(r.liquidated, 0);
    assert!(close(s.shares()[0] * 6.0, 100.0));
}

#[test]
fn portfolio_margin_recognizes_a_hedge() {
    let m = || {
        market(
            &[&[100.0, 3.0], &[100.0, 3.0], &[100.0, 3.0]],
            &[&[100.0, 3.0], &[100.0, 3.0], &[100.0, 3.0]],
            vec![
                Instrument::EQUITY,
                option(0, Right::Put, 95.0, 30, Settlement::Physical),
            ],
        )
    };
    let req = |margin: MarginModel| {
        let rules = Rules {
            margin,
            ..Rules::default()
        };
        let mut s = sim(m(), levered(10.0), rules);
        s.step_units(&[1_000.0, 10.0]).unwrap().margin_requirement
    };
    let pm = req(MarginModel::Portfolio(Portfolio::default()));
    let rt = req(MarginModel::RegT(RegT::default()));
    // Reg T: 25% of the stock plus the full premium; portfolio margin:
    // the worst loss with the put capping the downside near 5 + 3.
    assert!(close(rt, 25_000.0 + 3_000.0), "{rt}");
    assert!(pm < 0.5 * rt && pm > 0.0, "{pm}");
}

#[test]
fn option_costs_and_lots() {
    let m = market(
        &[&[100.0, 2.0], &[100.0, 2.0]],
        &[&[100.0, 2.0], &[100.0, 2.0]],
        vec![
            Instrument::EQUITY,
            option(0, Right::Call, 100.0, 5, Settlement::Physical),
        ],
    );
    let costs = Costs {
        option_commission: 0.65,
        option_half_spread_bps: 100.0,
        option_min_half_spread: 0.05,
        ..free()
    };
    let mut s = Simulator::new(Arc::new(m), costs, levered(1.0), 0, CAP).unwrap();
    // Option weights are notional: half of equity at 100 x 100 of stock per
    // contract is 5 contracts (whatever the premium).
    let r = s.step(&[0.0, 0.5]).unwrap();
    assert_eq!(s.shares()[1], 5.0);
    assert!(close(r.commission, 5.0 * 0.65));
    // The half-spread floor (0.05 > 1% of 2.00) per unit, x 100 x 5.
    assert!(close(r.spread_cost, 0.05 * 100.0 * 5.0), "{r:?}");
}

#[test]
fn half_spread_array_overrides_bps() {
    let mut m = market(&[&[10.0], &[10.0]], &[&[10.0], &[10.0]], vec![]);
    m.half_spread = Some(vec![f64::NAN, 0.02]);
    let costs = Costs {
        half_spread_bps: 5.0,
        ..free()
    };
    let mut s = Simulator::new(Arc::new(m), costs, Constraints::default(), 0, 1_000.0).unwrap();
    let r = s.step(&[1.0]).unwrap();
    assert!(close(r.spread_cost, 100.0 * 0.02));
}

#[test]
fn invalid_instruments_are_rejected() {
    let ok = || {
        let mut m = market(
            &[&[100.0, 1.0], &[100.0, 1.0], &[100.0, 1.0]],
            &[&[100.0, 1.0], &[100.0, 1.0], &[100.0, 1.0]],
            vec![
                Instrument::EQUITY,
                option(0, Right::Call, 100.0, 1, Settlement::Physical),
            ],
        );
        expire(&mut m, 1, 1);
        m
    };
    assert!(ok().validate().is_ok());
    let mut m = ok();
    m.listed[2 * 2 + 1] = true; // listed after expiry
    assert!(m.validate().is_err());
    let mut m = ok();
    m.dividend[1] = 0.1; // options pay no dividends
    assert!(m.validate().is_err());
    let mut m = ok();
    m.instruments[1] = option(1, Right::Call, 100.0, 1, Settlement::Physical);
    assert!(m.validate().is_err()); // its own underlying
    let mut m = ok();
    m.instruments[0] = Instrument {
        kind: Kind::Index { broad_based: true },
        multiplier: 1.0,
        lot: 0.0,
    };
    assert!(m.validate().is_err()); // tradable index; and physical on an index
    let mut m = ok();
    m.instruments.pop();
    assert!(m.validate().is_err());
}

#[test]
fn option_weights_are_notional() {
    // The call's premium gaps from 2 to 3 overnight while the stock opens
    // at 101: both legs are sized from the stock's open.
    let m = market(
        &[&[100.0, 2.0], &[101.0, 3.0], &[102.0, 3.5]],
        &[&[100.0, 2.0], &[101.0, 3.0], &[102.0, 3.5]],
        vec![
            Instrument::EQUITY,
            option(0, Right::Call, 105.0, 5, Settlement::Physical),
        ],
    );
    let mut s = sim(m, levered(2.5), Rules::default());
    s.step(&[1.0, -1.0]).unwrap();
    let (stock, calls) = (s.shares()[0], s.shares()[1]);
    // Whole contracts covering the shares bought at the same open.
    assert_eq!(calls, -(stock / 100.0).round());
    assert!((stock - CAP / 101.0).abs() < 1e-6);
    let w = s.weights().to_vec();
    assert!(close(w[1], calls * 100.0 * 101.0 / s.equity()), "{w:?}");
}
