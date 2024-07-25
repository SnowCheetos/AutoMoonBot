use crate::nodes::*;

#[derive(Debug)]
pub struct Article {
    pub(super) title: String,
    pub(super) summary: String,
    pub(super) sentiment: f64,
    pub(super) publisher: String,
}

#[derive(Debug)]
pub struct Publisher {
    pub(super) name: String,
    pub(super) sentiments: TimeSeries<f64>,
}

#[derive(Debug)]
pub struct Company {
    pub(super) name: String,
    pub(super) symbols: HashSet<String>,
    pub(super) income_statement: TimeSeries<IncomeStatement>,
    pub(super) balance_sheet: TimeSeries<BalanceSheet>,
    pub(super) cash_flow: TimeSeries<CashFlow>,
    pub(super) earnings: TimeSeries<Earnings>,
}

#[derive(Debug)]
pub struct Currency {
    pub(super) symbol: String,
    pub(super) history: TimeSeries<PriceAggregate>,
}

#[derive(Debug)]
pub struct Equity {
    pub(super) symbol: String,
    pub(super) history: TimeSeries<PriceAggregate>,
}

#[derive(Debug)]
pub struct Indices {
    pub(super) symbol: String,
    pub(super) history: TimeSeries<PriceAggregate>,
}

#[derive(Debug)]
pub struct ETFs {
    pub(super) symbol: String,
    pub(super) history: TimeSeries<PriceAggregate>,
}

#[derive(Debug)]
pub struct Bonds {
    pub(super) symbol: String,
    pub(super) interest_rate: f64,
    pub(super) maturity: Instant,
    pub(super) history: TimeSeries<PriceAggregate>,
}

#[derive(Debug)]
pub struct Options {
    pub(super) contract_id: String,
    pub(super) underlying: String,
    pub(super) strike: f64,
    pub(super) expiration: Instant,
    pub(super) history: TimeSeries<OptionsAggregate>,
}

impl Article {
    pub fn new(title: String, summary: String, sentiment: f64, publisher: String) -> Self {
        Article {
            title,
            summary,
            sentiment,
            publisher,
        }
    }

    pub fn publisher(&self) -> &String {
        &self.publisher
    }
}

impl Publisher {
    pub fn new(name: String, capacity: usize) -> Self {
        Publisher {
            name,
            sentiments: TimeSeries::new(capacity),
        }
    }
}

impl Currency {
    pub fn new(symbol: String, capacity: usize) -> Self {
        Currency {
            symbol,
            history: TimeSeries::new(capacity),
        }
    }
}

impl Equity {
    pub fn new(symbol: String, capacity: usize) -> Self {
        Equity {
            symbol,
            history: TimeSeries::new(capacity),
        }
    }
}

impl Indices {
    pub fn new(symbol: String, capacity: usize) -> Self {
        Indices {
            symbol,
            history: TimeSeries::new(capacity),
        }
    }
}

impl ETFs {
    pub fn new(symbol: String, capacity: usize) -> Self {
        ETFs {
            symbol,
            history: TimeSeries::new(capacity),
        }
    }
}

impl Bonds {
    pub fn new(symbol: String, interest_rate: f64, maturity: Instant, capacity: usize) -> Self {
        Bonds {
            symbol,
            interest_rate,
            maturity,
            history: TimeSeries::new(capacity),
        }
    }
}

impl Options {
    pub fn new(
        contract_id: String,
        underlying: String,
        strike: f64,
        expiration: Instant,
        capacity: usize,
    ) -> Self {
        Options {
            contract_id,
            underlying,
            strike,
            expiration,
            history: TimeSeries::new(capacity),
        }
    }
}

impl Company {
    pub fn new(name: String, symbols: HashSet<String>, capacity: usize) -> Self {
        Company {
            name,
            symbols,
            income_statement: TimeSeries::new(capacity),
            balance_sheet: TimeSeries::new(capacity),
            cash_flow: TimeSeries::new(capacity),
            earnings: TimeSeries::new(capacity),
        }
    }
}
