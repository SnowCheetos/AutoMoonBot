//! Property tests for the execution simulator.

use std::sync::Arc;

use moonrs::backtest::{Constraints, Costs, Market, Simulator};
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
        ..Costs::default()
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
