use crate::edges::*;

pub mod static_relations {
    use super::*;

    pub mod defs {

        #[derive(Debug, Clone)]
        pub struct Mentioned {
            pub(in crate::edges) source: String,
            pub(in crate::edges) target: String,
            pub(in crate::edges) src_uuid: String,
            pub(in crate::edges) tgt_uuid: String,
            pub(in crate::edges) relevance: f64,
            pub(in crate::edges) sentiment: f64,
        }

        #[derive(Debug, Clone)]
        pub struct Composed {}
    }

    pub mod impls {
        use super::*;

        impl Mentioned {
            pub fn new(
                source: &Article,
                target: &StaticEvent,
                relevance: f64,
                sentiment: f64,
            ) -> Self {
                Mentioned {
                    source: source.cls().to_owned(),
                    target: target.cls().to_owned(),
                    src_uuid: source.uuid().to_owned(),
                    tgt_uuid: target.uuid().to_owned(),
                    relevance,
                    sentiment,
                }
            }
        }
    }
}

pub mod dynamic_relations {
    use super::*;

    pub mod defs {
        use super::*;
    }

    pub mod impls {
        use super::*;
    }
}
