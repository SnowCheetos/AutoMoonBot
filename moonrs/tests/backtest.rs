//! Property tests for the execution simulator.

use std::sync::Arc;

use moonrs::backtest::instrument::{
    FutureSpec, Instrument, Kind, OptionSpec, Settlement, SettlementPrice, Style,
};
use moonrs::backtest::margin::{MarginModel, Portfolio, RegT};
use moonrs::backtest::pricing::Right;
use moonrs::backtest::{Constraints, Costs, Market, Rules, Simulator};
use proptest::prelude::*;

fn market(t: usize, s: usize, px: &[f64]) -> Market {
    // px: [t, s] closes; opens = previous close * gap.
    let mut open = vec![0.0; t * s];
    for k in 0..t {
        for i in 0..s {
            let c = px[k * s + i];
            open[k * s + i] = if k == 0 {
                c
            } else {
                0.5 * (c + px[(k - 1) * s + i])
            };
        }
    }
    Market {
        periods: t,
        securities: s,
        open,
        close: px.to_vec(),
        adv_shares: vec![1e15; t * s],
        sigma: vec![0.02; t * s],
        tradable: vec![true; t * s],
        listed: vec![true; t * s],
        split: vec![1.0; t * s],
        dividend: vec![0.0; t * s],
        instruments: Vec::new(),
        half_spread: None,
        rate: None,
    }
}

fn free() -> Costs {
    Costs {
        commission_bps: 0.0,
        half_spread_bps: 0.0,
        impact_coef: 0.0,
        borrow_rate: 0.0,
        margin_rate: 0.0,
        cash_rate: 0.0,
        option_commission: 0.0,
        future_commission: 0.0,
        option_half_spread_bps: 0.0,
        option_min_half_spread: 0.0,
        ..Costs::default()
    }
}

fn levered() -> Constraints {
    Constraints {
        long_only: false,
        max_gross: 50.0,
        max_weight: 50.0,
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: std::env::var("PROPTEST_CASES").ok().and_then(|v| v.parse().ok()).unwrap_or(256),
        .. ProptestConfig::default()
    })]

    /// Frictionless equity equals the analytic value of the same trades,
    /// including "hold" (NaN) entries that keep shares and use up budget.
    #[test]
    fn frictionless_matches_reference(
        (t, s, px, w) in (3usize..12, 1usize..5).prop_flat_map(|(t, s)| (
            Just(t), Just(s),
            prop::collection::vec(1.0f64..200.0, t * s),
            prop::collection::vec(prop::collection::vec(-0.4f64..1.0, s), t - 1),
        ))
    ) {
        let m = market(t, s, &px);
        let mut sim = Simulator::new(Arc::new(m.clone()), free(), Constraints::default(), 0, 1e6).unwrap();
        let mut cash = 1e6;
        let mut shares = vec![0.0; s];
        for (k, raw) in w.iter().enumerate() {
            let wk: Vec<f64> = raw.iter().map(|&x| if x < 0.0 { f64::NAN } else { x }).collect();
            let o = &m.open[(k + 1) * s..(k + 2) * s];
            let eq_open: f64 = cash + shares.iter().zip(o).map(|(q, p)| q * p).sum::<f64>();
            let held: f64 = (0..s)
                .filter(|&i| wk[i].is_nan())
                .map(|i| (shares[i] * o[i]).abs())
                .sum::<f64>() / eq_open;
            let gross: f64 = wk.iter().filter(|x| !x.is_nan()).sum();
            let budget = (1.0 - held).max(0.0);
            let scale = if gross > budget { budget / gross } else { 1.0 };
            for i in 0..s {
                if wk[i].is_nan() {
                    continue;
                }
                let target = wk[i] * scale * eq_open / o[i];
                cash -= (target - shares[i]) * o[i];
                shares[i] = target;
            }
            let c = &m.close[(k + 1) * s..(k + 2) * s];
            let eq: f64 = cash + shares.iter().zip(c).map(|(q, p)| q * p).sum::<f64>();
            let r = sim.step(&wk).unwrap();
            prop_assert!((r.equity - eq).abs() <= 1e-6 * eq.abs().max(1.0), "{} vs {}", r.equity, eq);
            for (got, want) in sim.shares().iter().zip(&shares) {
                prop_assert!((got - want).abs() <= 1e-9 * want.abs().max(1.0));
            }
        }
    }

    /// Frictions can only cost money when following the same weights.
    #[test]
    fn costs_never_help(
        (t, s, px, w) in (3usize..12, 1usize..5).prop_flat_map(|(t, s)| (
            Just(t), Just(s),
            prop::collection::vec(1.0f64..200.0, t * s),
            prop::collection::vec(prop::collection::vec(0.0f64..1.0, s), t - 1),
        )),
        spread in 0.0f64..50.0,
        comm in 0.0f64..10.0,
        k in 0.0f64..1.0,
    ) {
        let m = Arc::new(market(t, s, &px));
        let costly = Costs { half_spread_bps: spread, commission_bps: comm, impact_coef: k, ..free() };
        let mut a = Simulator::new(m.clone(), free(), Constraints::default(), 0, 1e6).unwrap();
        let mut b = Simulator::new(m, costly, Constraints::default(), 0, 1e6).unwrap();
        for wk in &w {
            let (ra, rb) = (a.step(wk).unwrap(), b.step(wk).unwrap());
            prop_assert!(rb.equity <= ra.equity * (1.0 + 1e-12));
            prop_assert!(rb.commission >= 0.0 && rb.spread_cost >= 0.0 && rb.impact_cost >= 0.0);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: std::env::var("PROPTEST_CASES").ok().and_then(|v| v.parse().ok()).unwrap_or(256),
        .. ProptestConfig::default()
    })]

    /// Futures: equity moves by exactly the variation margin, which is the
    /// position before the trade times the gap to the open plus the
    /// position after it times the move to the close.
    #[test]
    fn futures_equity_is_the_sum_of_variation_margin(
        (t, px, q) in (3usize..15).prop_flat_map(|t| (
            Just(t),
            prop::collection::vec(50.0f64..150.0, t),
            prop::collection::vec(-5i32..=5, t - 1),
        ))
    ) {
        let mut m = market(t, 1, &px);
        m.instruments = vec![Instrument {
            kind: Kind::Future(FutureSpec {
                underlying: None,
                expiry: t + 10,
                settlement: SettlementPrice::Close,
                initial_margin: 0.1,
                maintenance_margin: 0.08,
            }),
            multiplier: 10.0,
            lot: 1.0,
        }];
        let mut sim = Simulator::new(Arc::new(m.clone()), free(), levered(), 0, 1e6).unwrap();
        let (mut held, mut eq) = (0.0, 1e6);
        for (k, &qk) in q.iter().enumerate() {
            let (c0, o1, c1) = (m.close[k], m.open[k + 1], m.close[k + 1]);
            let r = sim.step_units(&[f64::from(qk)]).unwrap();
            let vm = 10.0 * (held * (o1 - c0) + f64::from(qk) * (c1 - o1));
            eq += vm;
            held = f64::from(qk);
            prop_assert!((r.variation_margin - vm).abs() <= 1e-9 * eq, "{r:?}");
            prop_assert!((r.equity - eq).abs() <= 1e-9 * eq);
            prop_assert!((sim.cash() - eq).abs() <= 1e-9 * eq);
        }
    }

    /// Exercising at expiry never creates or destroys value: with the option
    /// closing at its intrinsic value, equity is unchanged across it.
    #[test]
    fn exercise_at_intrinsic_conserves_equity(
        strike in 50.0f64..150.0,
        spot in 50.0f64..150.0,
        call in any::<bool>(),
        q in prop::sample::select(vec![-3.0, -1.0, 1.0, 2.0]),
        cash_settled in any::<bool>(),
    ) {
        let right = if call { Right::Call } else { Right::Put };
        let intrinsic = right_intrinsic(right, spot, strike).max(0.01);
        let px = [100.0, 1.0, 100.0, 1.0, spot, intrinsic, spot, 1.0];
        let mut m = market(4, 2, &px);
        // Fix the opens: the option trades at 1 then closes at intrinsic.
        m.open = vec![100.0, 1.0, 100.0, 1.0, spot, intrinsic, spot, 1.0];
        let settlement = if cash_settled {
            Settlement::Cash(SettlementPrice::Close)
        } else {
            Settlement::Physical
        };
        m.instruments = vec![
            Instrument::EQUITY,
            Instrument {
                kind: Kind::Option(OptionSpec {
                    underlying: 0, right, strike, expiry: 2, style: Style::European,
                    settlement, deliverable: 100.0,
                }),
                multiplier: 100.0,
                lot: 1.0,
            },
        ];
        m.listed[3 * 2 + 1] = false;
        m.tradable[3 * 2 + 1] = false;
        let mut sim = Simulator::new(Arc::new(m), free(), levered(), 0, 1e6).unwrap();
        sim.step_units(&[0.0, q]).unwrap();
        let r = sim.step_units(&[f64::NAN, f64::NAN]).unwrap();
        // Value just before settlement: cash + option at its close.
        let before = 1e6 - q * 100.0 + q * 100.0 * intrinsic;
        let exact = right_intrinsic(right, spot, strike);
        let after = 1e6 - q * 100.0 + if exact >= 0.01 { q * 100.0 * exact } else { 0.0 };
        prop_assert!((r.equity - after).abs() <= 1e-6, "{r:?} {after}");
        prop_assert!((before - after).abs() <= q.abs() * 100.0 * 0.01 + 1e-6);
        prop_assert_eq!(sim.shares()[1], 0.0);
        if !cash_settled && exact >= 0.01 {
            let sign = if call { 1.0 } else { -1.0 };
            prop_assert_eq!(sim.shares()[0], sign * q * 100.0);
        }
    }

    /// Margin requirements are positively homogeneous: doubling every
    /// position doubles the requirement, under both models.
    #[test]
    fn margin_is_homogeneous(
        stock in -500.0f64..500.0,
        calls in -5i32..=5,
        puts in -5i32..=5,
        spot in 80.0f64..120.0,
    ) {
        for margin in [MarginModel::RegT(RegT::default()), MarginModel::Portfolio(Portfolio::default())] {
            let req = |k: f64| {
                let mut m = market(3, 3, &[100.0, 3.0, 3.0, 100.0, 3.0, 3.0, spot, 3.0, 3.0]);
                m.instruments = vec![
                    Instrument::EQUITY,
                    opt(Right::Call, 105.0),
                    opt(Right::Put, 95.0),
                ];
                let rules = Rules { margin: margin.clone(), ..Rules::default() };
                let mut sim = Simulator::new(Arc::new(m), free(), levered(), 0, 1e9)
                    .unwrap().with_rules(rules).unwrap();
                sim.step_units(&[k * stock, k * f64::from(calls), k * f64::from(puts)]).unwrap();
                sim.step_units(&[f64::NAN; 3]).unwrap().margin_requirement
            };
            let (one, two) = (req(1.0), req(2.0));
            prop_assert!((two - 2.0 * one).abs() <= 1e-6 * one.max(1.0), "{margin:?}: {one} {two}");
        }
    }
}

fn right_intrinsic(right: Right, spot: f64, strike: f64) -> f64 {
    right.intrinsic(spot, strike)
}

fn opt(right: Right, strike: f64) -> Instrument {
    Instrument {
        kind: Kind::Option(OptionSpec {
            underlying: 0,
            right,
            strike,
            expiry: 30,
            style: Style::American,
            settlement: Settlement::Physical,
            deliverable: 100.0,
        }),
        multiplier: 100.0,
        lot: 1.0,
    }
}
