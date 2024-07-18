use crate::nodes::*;

/// A group of nodes whose values do not change once instantiated.
/// For instance, a news article could be represented as a static entity,
/// once an article is published, its content never changes.
/// (*well, I'm not so sure about that these days...*)
pub mod static_entities {
    use super::*;

    /// A general event that occured at some instance in time. Static events
    /// are not continuous, meaning their affects on other nodes are determined
    /// solely by its initial occurance. For instance, the acquisition of a
    /// company by another is considered a static event.
    #[derive(Debug, Clone)]
    pub struct StaticEvent {
        pub occurance: Instant,
        pub sentiment: f64,
        pub description: String,
    }

    #[derive(Debug, Clone)]
    pub struct Article {
        pub published: Instant,
        pub url: String,
        pub sentiment: f64,
    }
}

/// As the name suggests, dynamic entities are a group of nodes whose values
/// do change after instantiation. An obvious example would be an equity, the prices
/// of which change very frequently. An less obvious example would be a news publisher,
/// one might be tempted to treat it as a static entity at first, but a publisher can
/// publish new articles, change its average sentiment, and hire / fire authors, thus
/// making it a dynamic entity.
pub mod dynamic_entities {
    use super::*;

    /// A general event that occured for some duration of time, may or may not
    /// be ongoing. Temporal events does not mean that its attribute can change,
    /// it simply implies that the event spans more than an instance in time.
    /// For instance, Covid-19 can be considered a temporal event, as it first
    /// occured in late 2019, and had significant affects on the world at large
    /// until late 2022.
    #[derive(Debug, Clone)]
    pub struct TemporalEvent {
        pub start: Instant,
        pub end: Option<Instant>,
        pub sentiment: f64,
        pub description: String,
    }

    #[derive(Debug, Clone)]
    pub struct Author {
        pub name: String,
        pub sentiment: f64,
        pub articles: HashSet<String>,
    }

    #[derive(Debug, Clone)]
    pub struct Publisher {
        pub name: String,
        pub sentiment: f64,
        pub authors: HashSet<String>,
        pub articles: HashSet<String>,
    }

    #[derive(Debug, Clone)]
    pub struct Region {
        pub name: String,
        pub population: usize,
        pub currency: String,
        pub holidays: Vec<Instant>,
        pub ngdp_per_cap: f64,
    }

    #[derive(Debug, Clone)]
    pub struct Exchange {
        pub name: String,
        pub region: Option<String>,
        pub reg_open: Option<Instant>,
        pub reg_close: Option<Instant>,
    }

    /// As one could imagine, it'll be quite difficult to obtain
    /// historical data on private companies, therefore, only publically
    /// traded companies are concerned here.
    #[derive(Debug, Clone)]
    pub struct Company {
        pub name: String,
        pub region: String,
        pub founded: Instant,
    }

    /// Although it's difficult to obtain historical data for specific
    /// aspects of an sector, it can be approximately represented by a
    /// combination of attributes such as specific market indices, revenue
    /// of associated companies, which are easier to collect.
    #[derive(Debug, Clone)]
    pub struct Sector {
        pub name: String,
    }

    /// Named `Indices` to avoid conflict with other `Index` definitions.
    ///
    /// There are tons of market indices out there, tracking specific industries,
    /// sectors, companies, or any combination of assets. Thus technically, indices
    /// should be categorized as derivatives, Indices are not merely the sum of
    /// their components, index providers uses their own algorithms to come up
    /// with the index values, it's difficult to obtain the exact composition and
    /// methodology for an index, let alone its historical changes. For this reason,
    /// indices are treated as unique entities, which is not far from reality. In
    /// practice, indices affect their underlying components as much as they affect
    /// the indices themselves, especially for big names like NASDAQ and DOW Jones.
    #[derive(Debug, Clone)]
    pub struct Indices {
        pub symbol: String,
        pub region: String,
        pub buffer: TemporalDeque<Aggregate>,
    }

    #[derive(Debug, Clone)]
    pub struct Commodity {
        pub name: String,
        pub unit: String,
        pub regions: HashSet<String>,
    }

    #[derive(Debug, Clone)]
    pub struct Currency {
        pub symbol: String,
        pub region: String,
        pub floating: bool,
    }

    /// Although there are many different kinds of fixed-income assets,
    /// in practice, a majority of them can be categorized as `Bonds`.
    /// For instance, T-Bills, T-Notes, T-Bonds are fundamentally identical,
    /// albeit at different rates and maturity timelines.
    #[derive(Debug, Clone)]
    pub struct Bonds {
        pub symbol: String,
        pub region: String,
        pub interest_rate: Rate,
        pub buffer: TemporalDeque<Aggregate>,
    }

    #[derive(Debug, Clone)]
    pub struct Equity {
        pub symbol: String,
        pub region: String,
        pub exchanges: HashSet<String>,
        pub buffer: TemporalDeque<Aggregate>,
    }

    /// Ahh, crypto...
    #[derive(Debug, Clone)]
    pub struct Crypto {
        pub symbol: String,
        pub meme: bool,
        pub exchanges: HashSet<String>,
        pub buffer: TemporalDeque<Aggregate>,
    }
}

/// Entity derivatives are entites whose values are partially or entirely derived
/// from another entity. Since there wouldn't be much of a point in creating
/// derivatives from static entities whose value doesn't change (*it'll be like
/// taking a side bet on a fixed bet*), entity derivatives automatically imply
/// that both the derivative itself and its underlying entity are dynamic.
pub mod entity_derivatives {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct ETFs {
        pub symbol: String,
        pub underlying: String,
        pub buffer: TemporalDeque<Aggregate>,
    }

    #[derive(Debug, Clone)]
    pub struct Futures {
        pub symbol: String,
        pub underlying: String,
        pub expiration: Instant,
        pub tick_size: usize,
        pub contract_size: usize,
        pub contract_unit: String,
        pub initial_margin: f64,
        pub maintenance_margin: f64,
        pub physical_settlement: bool,
        pub exchanges: HashSet<String>,
    }

    /// Although theoretically you can short sell any asset, this is not valid in
    /// practice. For example, on paper I can short sell 500 tons of gold, but to
    /// do that I must borrow that 500 tons of gold from someone, if I could do that
    /// then I wouldn't be sitting here writing this. Rather, if I want to short
    /// gold, I would short sell the the equivalence of 500 tons of gold's worth of
    /// futures contracts, (*which I admit, is also not realistic*). In this case,
    /// the actual underlying asset is not gold, but rather the futures contracts,
    /// which itself is a marketable asset.
    ///
    /// Therefore, in this implementation, the underlying asset of a short sell must
    /// also be a marketable asset.
    #[derive(Debug, Clone)]
    pub struct Shorts {
        pub symbol: String,
        pub underlying: String,
        pub interest_rate: Rate,
        pub position_size: usize, // Fractional shorts are generally not allowed
    }

    #[derive(Debug, Clone)]
    pub struct Options {
        pub symbol: String,
        pub direction: Opt,
        pub underlying: String,
        pub expiration: Instant,
        pub strike_price: f64,
        pub contract_size: usize,
        pub exchanges: HashSet<String>,
    }
}

pub mod static_entities_impls {
    use super::*;

    impl StaticEvent {
        pub fn new(occurance: Instant, sentiment: f64, description: String) -> Self {
            StaticEvent {
                occurance,
                sentiment,
                description,
            }
        }
    }

    impl Article {
        pub fn new(published: Instant, url: String, sentiment: f64) -> Self {
            Article {
                published,
                url,
                sentiment,
            }
        }
    }
}

pub mod dynamic_entities_impls {
    use super::*;

    impl TemporalEvent {
        pub fn new(
            start: Instant,
            end: Option<Instant>,
            sentiment: f64,
            description: String,
        ) -> Self {
            TemporalEvent {
                start,
                end,
                sentiment,
                description,
            }
        }
    }

    impl Author {
        pub fn new(name: String, sentiment: f64) -> Self {
            Author {
                name,
                sentiment,
                articles: HashSet::new(),
            }
        }
    }

    impl Publisher {
        pub fn new(name: String, sentiment: f64) -> Self {
            Publisher {
                name,
                sentiment,
                authors: HashSet::new(),
                articles: HashSet::new(),
            }
        }
    }

    impl Region {
        pub fn new(
            name: String,
            population: usize,
            currency: String,
            holidays: Vec<Instant>,
            ngdp_per_cap: f64,
        ) -> Self {
            Region {
                name,
                population,
                currency,
                holidays,
                ngdp_per_cap,
            }
        }
    }

    impl Exchange {
        pub fn new(
            name: String,
            region: Option<String>,
            reg_open: Option<Instant>,
            reg_close: Option<Instant>,
        ) -> Self {
            Exchange {
                name,
                region,
                reg_open,
                reg_close,
            }
        }
    }

    impl Company {
        pub fn new(name: String, region: String, founded: Instant) -> Self {
            Company {
                name,
                region,
                founded,
            }
        }
    }

    impl Sector {
        pub fn new(name: String) -> Self {
            Sector { name }
        }
    }

    impl Indices {
        pub fn new(symbol: String, region: String, capacity: usize) -> Self {
            Indices {
                symbol,
                region,
                buffer: TemporalDeque::new(capacity),
            }
        }
    }

    impl Commodity {
        pub fn new(name: String, unit: String, regions: Option<HashSet<String>>) -> Self {
            Commodity {
                name,
                unit,
                regions: regions.unwrap_or_default(),
            }
        }
    }

    impl Currency {
        pub fn new(symbol: String, region: String, floating: bool) -> Self {
            Currency {
                symbol,
                region,
                floating,
            }
        }
    }

    impl Bonds {
        pub fn new(symbol: String, region: String, interest_rate: Rate, capacity: usize) -> Self {
            Bonds {
                symbol,
                region,
                interest_rate,
                buffer: TemporalDeque::new(capacity),
            }
        }
    }

    impl Equity {
        pub fn new(
            capacity: usize,
            symbol: String,
            region: String,
            exchanges: Vec<String>,
        ) -> Self {
            Equity {
                symbol,
                region,
                exchanges: HashSet::from_iter(exchanges.into_iter()),
                buffer: TemporalDeque::new(capacity),
            }
        }
    }

    impl Crypto {
        pub fn new(symbol: String, exchanges: HashSet<String>, capacity: usize) -> Self {
            let is_meme = !SERIOUS_CRYPTOS.contains(symbol.as_str());
            Crypto {
                symbol,
                meme: is_meme,
                exchanges,
                buffer: TemporalDeque::new(capacity),
            }
        }
    }
}

pub mod entity_derivatives_impls {
    use super::*;

    impl ETFs {
        pub fn new(symbol: String, underlying: String, capacity: usize) -> Self {
            ETFs {
                symbol,
                underlying,
                buffer: TemporalDeque::new(capacity),
            }
        }
    }

    impl Futures {
        pub fn new(
            symbol: String,
            underlying: String,
            expiration: Instant,
            tick_size: usize,
            contract_size: usize,
            contract_unit: String,
            initial_margin: f64,
            maintenance_margin: f64,
            physical_settlement: bool,
            exchanges: HashSet<String>,
        ) -> Self {
            Futures {
                symbol,
                underlying,
                expiration,
                tick_size,
                contract_size,
                contract_unit,
                initial_margin,
                maintenance_margin,
                physical_settlement,
                exchanges,
            }
        }
    }

    impl Shorts {
        pub fn new(
            symbol: String,
            underlying: String,
            interest_rate: Rate,
            position_size: usize,
        ) -> Self {
            Shorts {
                symbol,
                underlying,
                interest_rate,
                position_size,
            }
        }
    }

    impl Options {
        pub fn new(
            symbol: String,
            direction: Opt,
            underlying: String,
            expiration: Instant,
            strike_price: f64,
            exchanges: HashSet<String>,
            contract_size: Option<usize>,
        ) -> Self {
            Options {
                symbol,
                direction,
                underlying,
                expiration,
                strike_price,
                exchanges,
                contract_size: contract_size.unwrap_or(100),
            }
        }
    }
}
