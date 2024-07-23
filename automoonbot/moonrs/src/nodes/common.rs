use crate::nodes::*;

pub struct Article {
    pub(super) title: String,
    pub(super) summary: String,
    pub(super) sentiment: f64,
    pub(super) authors: HashSet<String>,
    pub(super) publisher: String,
}

pub struct Company {
    pub(super) name: String,
    pub(super) symbols: HashSet<String>,
    pub(super) sectors: HashSet<String>,
}

pub struct Sector {
    pub(super) name: String,
}

pub struct Currency {
    pub(super) symbol: String,
}

pub struct Bond {
    pub(super) symbol: String,
}

pub struct Equity {
    pub(super) symbol: String,
}

pub struct Indices {
    pub(super) symbol: String,
}

pub struct ETFs {
    pub(super) symbol: String,
}

pub struct Options {
    pub(super) symbol: String,
}