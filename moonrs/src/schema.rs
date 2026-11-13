//! Runtime-defined, append-only graph schema.
//!
//! The previous version hard-coded node and edge types (and Alpha Vantage's
//! statement layouts) as Rust enums. New relation types are expected to come
//! out of offline enrichment (supplier/customer, competitor, subsidiary, ...),
//! so the schema is data: node kinds, their feature channels and relation
//! types are declared at runtime, validated once, and can only be extended,
//! never changed. Re-declaring an identical item is a no-op so ingestion
//! scripts are idempotent.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Maximum number of node kinds / relations / channels per kind. These bound
/// the id types and keep exported dictionaries reasonable.
pub const MAX_KINDS: usize = u16::MAX as usize;
/// Upper bound on a single feature vector width (e.g. an embedding).
pub const MAX_DIM: usize = 1 << 16;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct KindId(pub u16);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RelationId(pub u16);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ChannelId(pub u16);

/// How a node kind exists in time.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeClass {
    /// Long-lived things (companies, securities, publishers). A snapshot
    /// contains every entity whose lifetime covers the valid time.
    Entity,
    /// Point-like occurrences (articles, filings). A snapshot contains only
    /// events that started within a trailing window, which keeps snapshot
    /// size bounded no matter how much history accumulates. `window_ns` is
    /// the default window and can be overridden per snapshot.
    Event { window_ns: i64 },
}

/// Shape and admissible values of a fixed-width `f32` vector: a node feature
/// channel or a relation's edge attributes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ValueSpec {
    pub dim: usize,
    /// Whether individual elements may be NaN, meaning "not reported". When
    /// false every element must be finite.
    pub nullable: bool,
    /// Optional inclusive bounds every non-null element must satisfy.
    pub bounds: Option<(f32, f32)>,
}

impl ValueSpec {
    pub fn new(dim: usize, nullable: bool, bounds: Option<(f32, f32)>) -> Result<Self> {
        if dim > MAX_DIM {
            return Err(Error::Schema(format!("dimension {dim} exceeds {MAX_DIM}")));
        }
        if let Some((lo, hi)) = bounds
            && !(lo.is_finite() && hi.is_finite() && lo <= hi)
        {
            return Err(Error::Schema(format!(
                "bounds ({lo}, {hi}) must be finite with lo <= hi"
            )));
        }
        Ok(ValueSpec {
            dim,
            nullable,
            bounds,
        })
    }

    /// Validate one vector. Infinities are always rejected: they are never a
    /// legitimate measurement and poison every downstream reduction.
    pub fn validate(&self, values: &[f32], context: impl Fn() -> String) -> Result<()> {
        if values.len() != self.dim {
            return Err(Error::Dimension {
                context: context(),
                expected: self.dim,
                got: values.len(),
            });
        }
        for (index, &value) in values.iter().enumerate() {
            let reason = if value.is_nan() {
                if self.nullable {
                    continue;
                }
                "NaN in a non-nullable vector"
            } else if value.is_infinite() {
                "infinite"
            } else if let Some((lo, hi)) = self.bounds {
                if value < lo || value > hi {
                    "out of declared bounds"
                } else {
                    continue;
                }
            } else {
                continue;
            };
            return Err(Error::InvalidValue {
                context: context(),
                index,
                value,
                reason,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChannelSpec {
    pub name: String,
    pub value: ValueSpec,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeKindSpec {
    pub name: String,
    pub class: NodeClass,
    pub channels: Vec<ChannelSpec>,
}

impl NodeKindSpec {
    pub fn channel_id(&self, name: &str) -> Option<ChannelId> {
        self.channels
            .iter()
            .position(|c| c.name == name)
            .map(|i| ChannelId(i as u16))
    }

    pub fn channel(&self, id: ChannelId) -> &ChannelSpec {
        &self.channels[id.0 as usize]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RelationSpec {
    pub name: String,
    pub src: KindId,
    pub dst: KindId,
    pub attrs: ValueSpec,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Schema {
    kinds: Vec<NodeKindSpec>,
    relations: Vec<RelationSpec>,
    #[serde(skip)]
    kind_by_name: HashMap<String, KindId>,
    #[serde(skip)]
    relation_by_name: HashMap<String, RelationId>,
}

/// Names become dictionary keys and PyG node/edge type names, so restrict
/// them to a conservative identifier alphabet.
pub fn validate_name(name: &str) -> Result<()> {
    let fail = |reason| {
        Err(Error::InvalidName {
            name: name.to_owned(),
            reason,
        })
    };
    if name.is_empty() || name.len() > 64 {
        return fail("must be 1..=64 bytes");
    }
    let mut chars = name.chars();
    let first = chars.next().expect("non-empty");
    if !first.is_ascii_lowercase() {
        return fail("must start with a lowercase ASCII letter");
    }
    if !chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') {
        return fail("may only contain lowercase ASCII letters, digits and '_'");
    }
    Ok(())
}

impl Schema {
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare a node kind. Identical re-declaration returns the existing id.
    pub fn define_kind(&mut self, spec: NodeKindSpec) -> Result<KindId> {
        validate_name(&spec.name)?;
        if let NodeClass::Event { window_ns } = spec.class
            && window_ns <= 0
        {
            return Err(Error::Schema(format!(
                "event kind {:?} needs a positive default window",
                spec.name
            )));
        }
        let mut seen = std::collections::HashSet::new();
        for c in &spec.channels {
            validate_name(&c.name)?;
            if !seen.insert(c.name.as_str()) {
                return Err(Error::Schema(format!(
                    "kind {:?} declares channel {:?} twice",
                    spec.name, c.name
                )));
            }
            ValueSpec::new(c.value.dim, c.value.nullable, c.value.bounds)?;
            if c.value.dim == 0 {
                return Err(Error::Schema(format!(
                    "channel {:?} of {:?} must have dim >= 1",
                    c.name, spec.name
                )));
            }
        }
        if spec.channels.len() > MAX_KINDS {
            return Err(Error::Capacity("too many channels".into()));
        }
        if let Some(&id) = self.kind_by_name.get(&spec.name) {
            return if self.kinds[id.0 as usize] == spec {
                Ok(id)
            } else {
                Err(Error::Redefinition {
                    what: "node kind",
                    name: spec.name,
                })
            };
        }
        if self.kinds.len() >= MAX_KINDS {
            return Err(Error::Capacity("too many node kinds".into()));
        }
        let id = KindId(self.kinds.len() as u16);
        self.kind_by_name.insert(spec.name.clone(), id);
        self.kinds.push(spec);
        Ok(id)
    }

    /// Declare a relation `src --name--> dst`. Identical re-declaration
    /// returns the existing id.
    pub fn define_relation(
        &mut self,
        name: &str,
        src: &str,
        dst: &str,
        attrs: ValueSpec,
    ) -> Result<RelationId> {
        validate_name(name)?;
        let attrs = ValueSpec::new(attrs.dim, attrs.nullable, attrs.bounds)?;
        let spec = RelationSpec {
            name: name.to_owned(),
            src: self.kind_id(src)?,
            dst: self.kind_id(dst)?,
            attrs,
        };
        if let Some(&id) = self.relation_by_name.get(name) {
            return if self.relations[id.0 as usize] == spec {
                Ok(id)
            } else {
                Err(Error::Redefinition {
                    what: "relation",
                    name: name.to_owned(),
                })
            };
        }
        if self.relations.len() >= MAX_KINDS {
            return Err(Error::Capacity("too many relations".into()));
        }
        let id = RelationId(self.relations.len() as u16);
        self.relation_by_name.insert(name.to_owned(), id);
        self.relations.push(spec);
        Ok(id)
    }

    pub fn kind_id(&self, name: &str) -> Result<KindId> {
        self.kind_by_name
            .get(name)
            .copied()
            .ok_or_else(|| Error::Unknown {
                what: "node kind",
                name: name.to_owned(),
            })
    }

    pub fn relation_id(&self, name: &str) -> Result<RelationId> {
        self.relation_by_name
            .get(name)
            .copied()
            .ok_or_else(|| Error::Unknown {
                what: "relation",
                name: name.to_owned(),
            })
    }

    pub fn channel_id(&self, kind: KindId, name: &str) -> Result<ChannelId> {
        let spec = self.kind(kind);
        spec.channel_id(name).ok_or_else(|| Error::Unknown {
            what: "channel",
            name: format!("{}.{}", spec.name, name),
        })
    }

    pub fn kind(&self, id: KindId) -> &NodeKindSpec {
        &self.kinds[id.0 as usize]
    }

    pub fn relation(&self, id: RelationId) -> &RelationSpec {
        &self.relations[id.0 as usize]
    }

    pub fn kinds(&self) -> &[NodeKindSpec] {
        &self.kinds
    }

    pub fn relations(&self) -> &[RelationSpec] {
        &self.relations
    }

    /// Rebuild lookup tables and re-validate everything; used after
    /// deserialization so a tampered file cannot produce an inconsistent
    /// schema.
    pub(crate) fn rebuild(kinds: Vec<NodeKindSpec>, relations: Vec<RelationSpec>) -> Result<Self> {
        let mut schema = Schema::new();
        for k in kinds {
            schema.define_kind(k)?;
        }
        for r in relations {
            let (src, dst) = (r.src.0 as usize, r.dst.0 as usize);
            if src >= schema.kinds.len() || dst >= schema.kinds.len() {
                return Err(Error::Format(format!(
                    "relation {:?} references a missing kind",
                    r.name
                )));
            }
            let (src, dst) = (
                schema.kinds[src].name.clone(),
                schema.kinds[dst].name.clone(),
            );
            schema.define_relation(&r.name, &src, &dst, r.attrs)?;
        }
        Ok(schema)
    }

    pub(crate) fn to_parts(&self) -> (Vec<NodeKindSpec>, Vec<RelationSpec>) {
        (self.kinds.clone(), self.relations.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kind(name: &str) -> NodeKindSpec {
        NodeKindSpec {
            name: name.into(),
            class: NodeClass::Entity,
            channels: vec![ChannelSpec {
                name: "x".into(),
                value: ValueSpec::new(2, false, None).unwrap(),
            }],
        }
    }

    #[test]
    fn names_are_restricted() {
        assert!(validate_name("equity").is_ok());
        assert!(validate_name("rev_supplies_2").is_ok());
        for bad in ["", "Equity", "1abc", "a-b", "a b", &"a".repeat(65)] {
            assert!(validate_name(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn redefinition_rules() {
        let mut s = Schema::new();
        let a = s.define_kind(kind("company")).unwrap();
        assert_eq!(s.define_kind(kind("company")).unwrap(), a);
        let mut different = kind("company");
        different.channels[0].value.dim = 3;
        assert!(matches!(
            s.define_kind(different),
            Err(Error::Redefinition { .. })
        ));
        let r = s
            .define_relation(
                "supplies",
                "company",
                "company",
                ValueSpec::new(1, false, None).unwrap(),
            )
            .unwrap();
        assert_eq!(
            s.define_relation(
                "supplies",
                "company",
                "company",
                ValueSpec::new(1, false, None).unwrap()
            )
            .unwrap(),
            r
        );
        assert!(
            s.define_relation(
                "supplies",
                "company",
                "company",
                ValueSpec::new(2, false, None).unwrap()
            )
            .is_err()
        );
        assert!(
            s.define_relation(
                "x",
                "company",
                "missing",
                ValueSpec::new(0, false, None).unwrap()
            )
            .is_err()
        );
    }

    #[test]
    fn value_validation() {
        let v = ValueSpec::new(3, false, Some((-1.0, 1.0))).unwrap();
        let ctx = || "t".to_string();
        assert!(v.validate(&[0.0, 1.0, -1.0], ctx).is_ok());
        assert!(v.validate(&[0.0, 1.0], ctx).is_err());
        assert!(v.validate(&[0.0, 1.5, 0.0], ctx).is_err());
        assert!(v.validate(&[0.0, f32::NAN, 0.0], ctx).is_err());
        assert!(v.validate(&[0.0, f32::INFINITY, 0.0], ctx).is_err());
        let n = ValueSpec::new(2, true, None).unwrap();
        assert!(n.validate(&[f32::NAN, 3.0], ctx).is_ok());
        assert!(n.validate(&[f32::NEG_INFINITY, 3.0], ctx).is_err());
        assert!(ValueSpec::new(1, false, Some((1.0, 0.0))).is_err());
        assert!(ValueSpec::new(1, false, Some((f32::NAN, 0.0))).is_err());
    }

    #[test]
    fn event_kinds_need_window() {
        let mut s = Schema::new();
        let mut k = kind("article");
        k.class = NodeClass::Event { window_ns: 0 };
        assert!(s.define_kind(k).is_err());
    }
}
