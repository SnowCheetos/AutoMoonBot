//! Option values for margin scenarios: the generalized Black–Scholes model
//! with a cost of carry `b` (`b = r` for options on a stock or an index,
//! `b = 0` for options on a future, i.e. Black-76), and its inverse, the
//! implied volatility.
//!
//! American options are valued as `max(European, intrinsic)`, the standard
//! lower bound. Volatilities are implied from the same European formula,
//! so for an American put trading above its European value the implied
//! volatility is slightly high; revaluing with it recovers the market price
//! at the current spot, which is what a scenario grid needs.

/// Call or put.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Right {
    Call,
    Put,
}

impl Right {
    /// Intrinsic value per unit of the underlying.
    #[inline]
    pub fn intrinsic(self, spot: f64, strike: f64) -> f64 {
        match self {
            Right::Call => (spot - strike).max(0.0),
            Right::Put => (strike - spot).max(0.0),
        }
    }
}

/// Smallest and largest volatilities the inversion searches.
pub const VOL_MIN: f64 = 1e-4;
pub const VOL_MAX: f64 = 8.0;

#[inline]
fn norm_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x * std::f64::consts::FRAC_1_SQRT_2)
}

#[inline]
fn norm_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Model inputs other than the volatility.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contract {
    pub right: Right,
    pub spot: f64,
    pub strike: f64,
    /// Years to expiry.
    pub tau: f64,
    /// Continuously compounded risk-free rate.
    pub rate: f64,
    /// Cost of carry.
    pub carry: f64,
}

impl Contract {
    /// European value at volatility `vol`; at `tau <= 0` the intrinsic value.
    pub fn european(&self, vol: f64) -> f64 {
        let Contract {
            right,
            spot: s,
            strike: k,
            tau,
            rate: r,
            carry: b,
        } = *self;
        if tau <= 0.0 {
            return right.intrinsic(s, k);
        }
        let df = (-r * tau).exp();
        let fwd_df = ((b - r) * tau).exp();
        let sd = vol * tau.sqrt();
        if sd.is_nan() || sd <= 0.0 {
            // Zero volatility: the discounted intrinsic value of the forward.
            return match right {
                Right::Call => (s * fwd_df - k * df).max(0.0),
                Right::Put => (k * df - s * fwd_df).max(0.0),
            };
        }
        let d1 = ((s / k).ln() + (b + 0.5 * vol * vol) * tau) / sd;
        let d2 = d1 - sd;
        match right {
            Right::Call => s * fwd_df * norm_cdf(d1) - k * df * norm_cdf(d2),
            Right::Put => k * df * norm_cdf(-d2) - s * fwd_df * norm_cdf(-d1),
        }
    }

    /// Value at `vol`, floored at intrinsic for an American option.
    #[inline]
    pub fn value(&self, vol: f64, american: bool) -> f64 {
        let v = self.european(vol);
        if american {
            v.max(self.right.intrinsic(self.spot, self.strike))
        } else {
            v
        }
    }

    fn vega(&self, vol: f64) -> f64 {
        let sd = vol * self.tau.sqrt();
        let d1 = ((self.spot / self.strike).ln() + (self.carry + 0.5 * vol * vol) * self.tau) / sd;
        self.spot * ((self.carry - self.rate) * self.tau).exp() * norm_pdf(d1) * self.tau.sqrt()
    }

    /// Volatility at which the European value equals `price`, if one exists
    /// in `[VOL_MIN, VOL_MAX]`. `None` when the price is outside the
    /// no-arbitrage range the model can reach, or `tau <= 0`.
    ///
    /// Safeguarded Newton: a Newton step that leaves the current bracket is
    /// replaced by bisection, so it converges for any price in range.
    pub fn implied_vol(&self, price: f64) -> Option<f64> {
        if !(self.tau > 0.0 && price.is_finite() && self.spot > 0.0 && self.strike > 0.0) {
            return None;
        }
        let (mut lo, mut hi) = (VOL_MIN, VOL_MAX);
        let (p_lo, p_hi) = (self.european(lo), self.european(hi));
        let tol = 1e-12 * self.spot.max(self.strike);
        if price < p_lo - tol || price > p_hi + tol {
            return None;
        }
        if price <= p_lo {
            return Some(lo);
        }
        if price >= p_hi {
            return Some(hi);
        }
        // Brenner–Subrahmanyam start, clamped into the bracket.
        let mut v =
            ((2.0 * std::f64::consts::PI / self.tau).sqrt() * price / self.spot).clamp(0.05, 2.0);
        for _ in 0..100 {
            let f = self.european(v) - price;
            if f.abs() <= tol {
                return Some(v);
            }
            if f > 0.0 {
                hi = v;
            } else {
                lo = v;
            }
            let vega = self.vega(v);
            let next = if vega > 0.0 { v - f / vega } else { f64::NAN };
            v = if next > lo && next < hi {
                next
            } else {
                0.5 * (lo + hi)
            };
            if hi - lo <= 1e-12 {
                return Some(v);
            }
        }
        Some(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(right: Right, spot: f64, strike: f64, tau: f64) -> Contract {
        Contract {
            right,
            spot,
            strike,
            tau,
            rate: 0.05,
            carry: 0.05,
        }
    }

    #[test]
    fn matches_reference_values() {
        // Hull, Options, Futures and Other Derivatives, example 15.6:
        // S=42, K=40, r=10%, sigma=20%, T=0.5 -> call 4.76, put 0.81.
        let mut call = c(Right::Call, 42.0, 40.0, 0.5);
        call.rate = 0.1;
        call.carry = 0.1;
        let put = Contract {
            right: Right::Put,
            ..call
        };
        assert!((call.european(0.2) - 4.759_422).abs() < 1e-5);
        assert!((put.european(0.2) - 0.808_599).abs() < 1e-5);
        // Put-call parity.
        let parity = call.european(0.3) - put.european(0.3) - (42.0 - 40.0 * (-0.05f64).exp());
        assert!(parity.abs() < 1e-12);
    }

    #[test]
    fn black76_is_zero_carry() {
        // Black-76 call on a future: F=20, K=20, r=9%, sigma=25%, T=4/12.
        let k = Contract {
            right: Right::Call,
            spot: 20.0,
            strike: 20.0,
            tau: 4.0 / 12.0,
            rate: 0.09,
            carry: 0.0,
        };
        // e^{-rT} (F N(d1) - K N(d2)) with d1 = sigma sqrt(T) / 2.
        let sd = 0.25 * (4.0f64 / 12.0).sqrt();
        let want =
            (-0.09f64 * 4.0 / 12.0).exp() * 20.0 * (norm_cdf(sd / 2.0) - norm_cdf(-sd / 2.0));
        assert!((k.european(0.25) - want).abs() < 1e-12);
    }

    #[test]
    fn implied_vol_round_trips() {
        for right in [Right::Call, Right::Put] {
            for &(s, k, tau) in &[
                (100.0, 100.0, 0.25),
                (100.0, 70.0, 1.0),
                (100.0, 130.0, 0.02),
            ] {
                for &vol in &[0.05, 0.2, 0.8, 3.0] {
                    let x = c(right, s, k, tau);
                    let p = x.european(vol);
                    // Deep out-of-the-money values below a tick carry no
                    // information about the volatility.
                    if p < 1e-6 {
                        continue;
                    }
                    let iv = x.implied_vol(p).unwrap();
                    assert!(
                        (x.european(iv) - p).abs() < 1e-9 * s,
                        "{right:?} {s} {k} {tau} {vol}: {iv}"
                    );
                }
            }
        }
    }

    #[test]
    fn implied_vol_rejects_arbitrage_and_expiry() {
        let x = c(Right::Call, 100.0, 90.0, 0.5);
        assert!(x.implied_vol(5.0).is_none()); // below intrinsic
        assert!(x.implied_vol(150.0).is_none()); // above the spot
        assert!(c(Right::Call, 100.0, 90.0, 0.0).implied_vol(10.0).is_none());
        assert_eq!(c(Right::Put, 100.0, 90.0, 0.0).european(0.3), 0.0);
    }

    #[test]
    fn american_floor() {
        // A deep in-the-money put with a high rate is worth less than its
        // intrinsic value as a European option.
        let mut x = c(Right::Put, 50.0, 100.0, 1.0);
        x.rate = 0.1;
        x.carry = 0.1;
        assert!(x.european(0.2) < 50.0);
        assert_eq!(x.value(0.2, true), 50.0);
    }
}
