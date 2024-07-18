use crate::nodes::*;

#[derive(Debug, Clone)]
pub struct Rate {
    rate: f64,
    unit: Duration,
}

/// A group of nodes whose values do not change once instantiated.
/// For instance, a news article could be represented as a static entity,
/// once an article is published, its content never changes.
/// (*well, I'm not so sure about that these days...*)
pub mod static_entities {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct Article {
        pub published: Instant,
    }

    #[derive(Debug, Clone)]
    pub struct Event {
        pub occurance: Instant,
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
        pub region: String,
        pub reg_open: Instant,
        pub reg_close: Instant,
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
        pub producers: HashSet<String>,
    }

    #[derive(Debug, Clone)]
    pub struct Currency {
        pub symbol: String,
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
}

pub mod entity_derivatives {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct ETFs<'a, T>
    where
        T: Asset,
    {
        pub symbol: String,
        pub underlying: &'a T,
        pub buffer: TemporalDeque<Aggregate>,
    }

    #[derive(Debug, Clone)]
    pub struct Futures<'a, T>
    where
        T: Asset,
    {
        pub symbol: String,
        pub underlying: &'a T,
        pub expiration: Instant,
        pub tick_size: usize,
        pub contract_size: usize,
        pub initial_margin: f64,
        pub maintenance_margin: f64,
        pub physical_settlement: bool,
        pub exchanges: HashSet<String>,
    }

    /// Although theoretically you can short sell any asset, this is not valid in
    /// practice. For example, on paper I can short sell 500 tons of gold, but to
    /// do that I must borrow that 500 tons of gold from someone, and I wouldn't be
    /// sitting here writing this if I could do that. Rather, if I want to short
    /// gold, I would short sell the futures contract of gold, an equivalence of
    /// 500 tons worth (*which is also not realistic...*). In this case, the actual
    /// underlying asset is not gold, but rather the futures contracts, which itself
    /// is a marketable asset.
    /// 
    /// Therefore, in this implementation, the underlying asset of a short sell must
    /// also be a marketable asset.
    #[derive(Debug, Clone)]
    pub struct Shorts<'a, T>
    where
        T: Marketable,
    {
        pub symbol: String,
        pub underlying: &'a T,
        pub position_size: f64,
    }

    #[derive(Debug, Clone)]
    pub struct Options<'a, T>
    where
        T: Marketable,
    {
        pub symbol: String,
        pub underlying: &'a T,
        pub expiration: Instant,
        pub direction: &'static str, // Can either be Call or Put
        pub strike: f64,
        pub contract_size: usize,
        pub exchanges: HashSet<String>,
    }
}

impl Equity {
    pub fn new(capacity: usize, symbol: String, region: String, exchanges: Vec<String>) -> Self {
        Equity {
            symbol,
            region,
            exchanges: HashSet::from_iter(exchanges.into_iter()),
            buffer: TemporalDeque::new(capacity),
        }
    }
}
