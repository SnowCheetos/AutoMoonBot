//! Property tests for the point-in-time guarantees.
//!
//! An independent, deliberately naive reference model recomputes every
//! snapshot from the raw fact log. The store must agree with it exactly, must
//! be causal (facts known after `known_at` cannot change a snapshot), must be
//! independent of ingestion order, and must round-trip through its file
//! format.

use std::collections::{BTreeMap, HashMap};

use moonrs::{
    ChannelSpec, MISSING_TIME, NodeClass, NodeKindSpec, Snapshot, SnapshotRequest, TemporalGraph,
    ValueSpec,
};
use proptest::prelude::*;

const ENTITY_KEYS: &[&str] = &["aapl", "msft", "nvda", "tsm"];
const EVENT_KEYS: &[&str] = &["a1", "a2", "a3", "a4", "a5"];
const WINDOW: i64 = 30;

#[derive(Clone, Debug)]
enum Fact {
    Node {
        kind: &'static str,
        key: &'static str,
        start: i64,
        end: i64,
        known: i64,
        assert: bool,
    },
    Obs {
        kind: &'static str,
        key: &'static str,
        valid: i64,
        known: i64,
        values: Vec<f32>,
    },
    Edge {
        rel: &'static str,
        src: &'static str,
        dst: &'static str,
        start: i64,
        end: i64,
        known: i64,
        attrs: Option<Vec<f32>>,
    },
}

impl Fact {
    fn known(&self) -> i64 {
        match self {
            Fact::Node { known, .. } | Fact::Obs { known, .. } | Fact::Edge { known, .. } => *known,
        }
    }
    fn set_known(&mut self, k: i64) {
        match self {
            Fact::Node { known, .. } | Fact::Obs { known, .. } | Fact::Edge { known, .. } => {
                *known = k
            }
        }
    }
}

fn graph() -> TemporalGraph {
    let mut g = TemporalGraph::new();
    g.define_kind(NodeKindSpec {
        name: "company".into(),
        class: NodeClass::Entity,
        channels: vec![ChannelSpec {
            name: "fund".into(),
            value: ValueSpec::new(2, true, None).unwrap(),
        }],
    })
    .unwrap();
    g.define_kind(NodeKindSpec {
        name: "article".into(),
        class: NodeClass::Event { window_ns: WINDOW },
        channels: vec![ChannelSpec {
            name: "emb".into(),
            value: ValueSpec::new(3, false, Some((-1.0, 1.0))).unwrap(),
        }],
    })
    .unwrap();
    g.define_relation(
        "supplies",
        "company",
        "company",
        ValueSpec::new(1, true, None).unwrap(),
    )
    .unwrap();
    g.define_relation(
        "mentions",
        "article",
        "company",
        ValueSpec::new(2, false, None).unwrap(),
    )
    .unwrap();
    // Declare every key up front (with a retraction that asserts nothing), so
    // facts can be ingested in any order.
    for (kind, keys) in [("company", ENTITY_KEYS), ("article", EVENT_KEYS)] {
        let n = keys.len();
        g.retract_nodes(
            kind,
            keys,
            &vec![-1_000_000; n],
            &vec![-999_999; n],
            &vec![-1_000_000; n],
        )
        .unwrap();
    }
    g
}

fn apply(g: &mut TemporalGraph, f: &Fact) -> moonrs::Result<moonrs::BatchStats> {
    match f {
        Fact::Node {
            kind,
            key,
            start,
            end,
            known,
            assert,
        } => {
            let args = (&[*key][..], &[*start][..], &[*end][..], &[*known][..]);
            if *assert {
                g.upsert_nodes(kind, args.0, args.1, args.2, args.3)
            } else {
                g.retract_nodes(kind, args.0, args.1, args.2, args.3)
            }
        }
        Fact::Obs {
            kind,
            key,
            valid,
            known,
            values,
        } => {
            let ch = if *kind == "company" { "fund" } else { "emb" };
            g.observe(kind, ch, &[*key], &[*valid], &[*known], values)
        }
        Fact::Edge {
            rel,
            src,
            dst,
            start,
            end,
            known,
            attrs,
        } => match attrs {
            Some(a) => g.assert_edges(rel, &[*src], &[*dst], &[*start], &[*end], &[*known], a),
            None => g.retract_edges(rel, &[*src], &[*dst], &[*start], &[*end], &[*known]),
        },
    }
}

// ------------------------------------------------------------- reference

struct Reference<'a> {
    facts: &'a [Fact],
}

impl Reference<'_> {
    /// Deciding interval fact for a key: the max (known, start, end) among
    /// facts known by `t_known` whose interval contains `t_valid`.
    fn decide<'f>(
        facts: impl Iterator<Item = (&'f Fact, i64, i64, i64)>,
        t_valid: i64,
        t_known: i64,
    ) -> Option<&'f Fact> {
        facts
            .filter(|&(_, s, e, k)| k <= t_known && s <= t_valid && t_valid < e)
            .max_by_key(|&(_, s, e, k)| (k, s, e))
            .map(|(f, ..)| f)
    }

    fn snapshot(&self, t_known: i64, t_valid: i64, hist: &HashMap<&str, usize>) -> Snapshot {
        let mut nodes = vec![];
        let mut rows: HashMap<(&str, &str), usize> = HashMap::new();
        for (kind, keys, ch, dim, nullable) in [
            ("company", ENTITY_KEYS, "fund", 2usize, true),
            ("article", EVENT_KEYS, "emb", 3usize, false),
        ] {
            let mut present: Vec<(&str, i64, i64)> = vec![];
            for &key in keys {
                let versions = self.facts.iter().filter_map(|f| match f {
                    Fact::Node {
                        kind: k,
                        key: kk,
                        start,
                        end,
                        known,
                        ..
                    } if *k == kind && *kk == key => Some((f, *start, *end, *known)),
                    _ => None,
                });
                if let Some(Fact::Node {
                    assert: true,
                    start,
                    known,
                    ..
                }) = Self::decide(versions, t_valid, t_known)
                {
                    if kind == "article" && *start < t_valid - WINDOW {
                        continue;
                    }
                    present.push((key, *start, *known));
                }
            }
            present.sort_by_key(|p| p.0);
            for (i, p) in present.iter().enumerate() {
                rows.insert((kind, p.0), i);
            }
            let k = *hist.get(kind).unwrap_or(&1);
            let n = present.len();
            let mut block = moonrs::ChannelBlock {
                name: ch.into(),
                dim,
                history: k,
                values: vec![0.0; n * k * dim],
                present: vec![0; n * k],
                element_present: nullable.then(|| vec![0; n * k * dim]),
                valid_time: vec![MISSING_TIME; n * k],
                known_time: vec![MISSING_TIME; n * k],
            };
            for (i, &(key, ..)) in present.iter().enumerate() {
                // Latest revision per valid time, known by t_known.
                let mut per_valid: BTreeMap<i64, (i64, &Vec<f32>)> = BTreeMap::new();
                for f in self.facts {
                    if let Fact::Obs {
                        kind: kk,
                        key: kkey,
                        valid,
                        known,
                        values,
                    } = f
                        && *kk == kind
                        && *kkey == key
                        && *valid <= t_valid
                        && *known <= t_known
                    {
                        let e = per_valid.entry(*valid).or_insert((*known, values));
                        if *known > e.0 {
                            *e = (*known, values);
                        }
                    }
                }
                for (j, (valid, (known, values))) in per_valid.iter().rev().take(k).enumerate() {
                    let slot = k - 1 - j;
                    let at = i * k + slot;
                    block.present[at] = 1;
                    block.valid_time[at] = *valid;
                    block.known_time[at] = *known;
                    for (e, &x) in values.iter().enumerate() {
                        block.values[at * dim + e] = if x.is_nan() { 0.0 } else { x };
                        if let Some(ep) = block.element_present.as_mut() {
                            ep[at * dim + e] = u8::from(!x.is_nan());
                        }
                    }
                }
            }
            nodes.push(moonrs::NodeBlock {
                kind: kind.into(),
                keys: present.iter().map(|p| p.0.to_string()).collect(),
                since: present.iter().map(|p| p.1).collect(),
                known_time: present.iter().map(|p| p.2).collect(),
                channels: vec![block],
            });
        }
        let mut edges = vec![];
        for (rel, sk, dk, sk_keys, dim, nullable) in [
            ("supplies", "company", "company", ENTITY_KEYS, 1usize, true),
            ("mentions", "article", "company", EVENT_KEYS, 2usize, false),
        ] {
            let mut found = vec![];
            for &s in sk_keys {
                for &d in ENTITY_KEYS {
                    let (Some(&rs), Some(&rd)) = (rows.get(&(sk, s)), rows.get(&(dk, d))) else {
                        continue;
                    };
                    let versions = self.facts.iter().filter_map(|f| match f {
                        Fact::Edge {
                            rel: r,
                            src,
                            dst,
                            start,
                            end,
                            known,
                            ..
                        } if *r == rel && *src == s && *dst == d => Some((f, *start, *end, *known)),
                        _ => None,
                    });
                    if let Some(Fact::Edge {
                        attrs: Some(a),
                        start,
                        known,
                        ..
                    }) = Self::decide(versions, t_valid, t_known)
                    {
                        found.push((rs as i64, rd as i64, *start, *known, a.clone()));
                    }
                }
            }
            found.sort_by_key(|f| (f.0, f.1));
            edges.push(moonrs::EdgeBlock {
                relation: rel.into(),
                src_kind: sk.into(),
                dst_kind: dk.into(),
                src: found.iter().map(|f| f.0).collect(),
                dst: found.iter().map(|f| f.1).collect(),
                attr_dim: dim,
                attrs: found
                    .iter()
                    .flat_map(|f| f.4.iter().map(|x| if x.is_nan() { 0.0 } else { *x }))
                    .collect(),
                attr_present: nullable.then(|| {
                    found
                        .iter()
                        .flat_map(|f| f.4.iter().map(|x| u8::from(!x.is_nan())))
                        .collect()
                }),
                since: found.iter().map(|f| f.2).collect(),
                known_time: found.iter().map(|f| f.3).collect(),
            });
        }
        Snapshot {
            known_at: t_known,
            valid_at: t_valid,
            nodes,
            edges,
        }
    }
}

// ------------------------------------------------------------- strategies

fn time() -> impl Strategy<Value = i64> {
    0i64..100
}

fn interval() -> impl Strategy<Value = (i64, i64)> {
    (time(), 1i64..60, any::<bool>()).prop_map(
        |(s, len, open)| {
            if open { (s, i64::MAX) } else { (s, s + len) }
        },
    )
}

fn nullable_val() -> impl Strategy<Value = f32> {
    prop_oneof![4 => -100.0f32..100.0, 1 => Just(f32::NAN)]
}

fn fact() -> impl Strategy<Value = Fact> {
    let ent = prop::sample::select(ENTITY_KEYS);
    let evt = prop::sample::select(EVENT_KEYS);
    prop_oneof![
        (ent.clone(), interval(), time(), prop::bool::weighted(0.75)).prop_map(
            |(key, (start, end), known, assert)| Fact::Node {
                kind: "company",
                key,
                start,
                end,
                known,
                assert
            }
        ),
        (evt.clone(), time(), time(), prop::bool::weighted(0.85)).prop_map(
            |(key, start, known, assert)| Fact::Node {
                kind: "article",
                key,
                start,
                end: i64::MAX,
                known,
                assert
            }
        ),
        (
            ent.clone(),
            time(),
            time(),
            prop::collection::vec(nullable_val(), 2)
        )
            .prop_map(|(key, valid, known, values)| Fact::Obs {
                kind: "company",
                key,
                valid,
                known,
                values
            }),
        (
            evt.clone(),
            time(),
            time(),
            prop::collection::vec(-1.0f32..=1.0, 3)
        )
            .prop_map(|(key, valid, known, values)| Fact::Obs {
                kind: "article",
                key,
                valid,
                known,
                values
            }),
        (
            ent.clone(),
            ent.clone(),
            interval(),
            time(),
            prop::option::weighted(0.8, nullable_val())
        )
            .prop_map(|(src, dst, (start, end), known, a)| Fact::Edge {
                rel: "supplies",
                src,
                dst,
                start,
                end,
                known,
                attrs: a.map(|x| vec![x])
            }),
        (
            evt,
            ent,
            interval(),
            time(),
            prop::option::weighted(0.9, (-1.0f32..1.0, 0.0f32..1.0))
        )
            .prop_map(|(src, dst, (start, end), known, a)| Fact::Edge {
                rel: "mentions",
                src,
                dst,
                start,
                end,
                known,
                attrs: a.map(|(x, y)| vec![x, y])
            }),
    ]
}

/// Fact logs whose knowledge times are unique (so no two facts can conflict)
/// plus some exact duplicates (to exercise idempotency).
fn log() -> impl Strategy<Value = Vec<Fact>> {
    (
        prop::collection::vec(fact(), 1..60),
        prop::collection::vec(any::<prop::sample::Index>(), 0..8),
    )
        .prop_map(|(mut facts, dups)| {
            for (i, f) in facts.iter_mut().enumerate() {
                let k = f.known() * 1000 + i as i64;
                f.set_known(k);
            }
            let extra: Vec<Fact> = dups.iter().map(|ix| ix.get(&facts).clone()).collect();
            facts.extend(extra);
            facts
        })
}

/// Knowledge/valid times are drawn adversarially: half the time exactly at,
/// just before, or just after some fact's knowledge time (where off-by-one
/// leaks live), otherwise uniformly.
fn probe(
    facts: &[Fact],
    pick: prop::sample::Index,
    delta: i64,
    uniform: i64,
    boundary: bool,
) -> i64 {
    if boundary {
        (pick.get(facts).known() + delta).max(0)
    } else {
        uniform
    }
}

fn build(facts: &[Fact]) -> TemporalGraph {
    let mut g = graph();
    for f in facts {
        apply(&mut g, f).expect("conflict-free log must ingest");
    }
    g
}

fn request(t_known: i64, t_valid: i64, hist: &HashMap<&str, usize>) -> SnapshotRequest {
    let mut r = SnapshotRequest::at(t_known);
    r.valid_at = Some(t_valid);
    if let Some(&k) = hist.get("company") {
        r.history.insert("company.fund".into(), k);
    }
    if let Some(&k) = hist.get("article") {
        r.history.insert("article.emb".into(), k);
    }
    r
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: std::env::var("PROPTEST_CASES").ok().and_then(|v| v.parse().ok()).unwrap_or(512),
        .. ProptestConfig::default()
    })]

    #[test]
    fn matches_reference_model(
        facts in log(),
        pick in any::<prop::sample::Index>(),
        delta in -1i64..=1,
        uniform in 0i64..110_000,
        boundary in any::<bool>(),
        dv in -20i64..20,
        hc in 1usize..4,
        ha in 1usize..3,
    ) {
        let t_known = probe(&facts, pick, delta, uniform, boundary);
        let g = build(&facts);
        let t_valid = (t_known / 1000 + dv).max(0);
        let hist: HashMap<&str, usize> = [("company", hc), ("article", ha)].into();
        let got = g.snapshot(&request(t_known, t_valid, &hist)).unwrap();
        let want = Reference { facts: &facts }.snapshot(t_known, t_valid, &hist);
        prop_assert_eq!(got, want);
    }

    #[test]
    fn future_facts_cannot_leak(
        facts in log(),
        pick in any::<prop::sample::Index>(),
        delta in -1i64..=1,
        uniform in 0i64..110_000,
        boundary in any::<bool>(),
        hc in 1usize..4,
    ) {
        let t_known = probe(&facts, pick, delta, uniform, boundary);
        let full = build(&facts);
        let past: Vec<Fact> = facts.iter().filter(|f| f.known() <= t_known).cloned().collect();
        let truncated = build(&past);
        let t_valid = t_known / 1000;
        let hist: HashMap<&str, usize> = [("company", hc)].into();
        let a = full.snapshot(&request(t_known, t_valid, &hist)).unwrap();
        let b = truncated.snapshot(&request(t_known, t_valid, &hist)).unwrap();
        // Every returned timestamp is knowable.
        for n in &a.nodes {
            prop_assert!(n.known_time.iter().all(|&k| k <= t_known));
            for c in &n.channels {
                prop_assert!(c.known_time.iter().all(|&k| k == MISSING_TIME || k <= t_known));
                prop_assert!(c.valid_time.iter().all(|&v| v == MISSING_TIME || v <= t_valid));
            }
        }
        for e in &a.edges {
            prop_assert!(e.known_time.iter().all(|&k| k <= t_known));
        }
        prop_assert_eq!(a, b);
    }

    #[test]
    fn ingestion_order_is_irrelevant(
        facts in log(),
        perm in any::<u64>(),
        t_known in 0i64..110_000,
    ) {
        let mut shuffled = facts.clone();
        // Deterministic Fisher-Yates from the drawn seed.
        let mut s = perm | 1;
        for i in (1..shuffled.len()).rev() {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            shuffled.swap(i, (s % (i as u64 + 1)) as usize);
        }
        let hist: HashMap<&str, usize> = [("company", 3)].into();
        let req = request(t_known, t_known / 1000, &hist);
        prop_assert_eq!(build(&facts).snapshot(&req).unwrap(), build(&shuffled).snapshot(&req).unwrap());
    }

    #[test]
    fn file_round_trip(facts in log(), t_known in 0i64..110_000) {
        let g = build(&facts);
        let bytes = g.to_bytes().unwrap();
        let h = TemporalGraph::from_bytes(&bytes).unwrap();
        let req = request(t_known, t_known / 1000, &[("company", 2)].into());
        prop_assert_eq!(g.snapshot(&req).unwrap(), h.snapshot(&req).unwrap());
        prop_assert_eq!(h.to_bytes().unwrap(), bytes);
    }

    #[test]
    fn failed_batches_change_nothing(facts in log(), t_known in 0i64..110_000) {
        let mut g = build(&facts);
        let req = request(t_known, t_known / 1000, &[("company", 3)].into());
        let before = g.snapshot(&req).unwrap();
        let before_count = g.observation_count("company", "fund").unwrap();
        // A batch of new observations ending in a row that conflicts with an
        // earlier row of the same batch.
        let err = g.observe(
            "company", "fund",
            &["aapl", "msft", "aapl"],
            &[5, 6, 5],
            &[999_999, 999_999, 999_999],
            &[1.0, 2.0, 3.0, 4.0, 1.5, 2.0],
        );
        prop_assert!(err.is_err());
        // A node batch that fails on an invalid key after creating new keys.
        let long = "x".repeat(2000);
        prop_assert!(g.upsert_nodes("company", &["new1", long.as_str()], &[0, 0], &[10, 10], &[1, 1]).is_err());
        prop_assert!(g.node_count("company").unwrap() == ENTITY_KEYS.len());
        prop_assert_eq!(g.observation_count("company", "fund").unwrap(), before_count);
        prop_assert_eq!(g.snapshot(&req).unwrap(), before);
    }
}

// ------------------------------------------------------------- examples

#[test]
fn restatement_is_invisible_before_it_is_known() {
    let mut g = graph();
    g.upsert_nodes("company", &["aapl"], &[0], &[i64::MAX], &[0])
        .unwrap();
    // Q1 revenue 10 reported at t=100, restated to 8 at t=200.
    g.observe(
        "company",
        "fund",
        &["aapl", "aapl"],
        &[90, 90],
        &[100, 200],
        &[10.0, 1.0, 8.0, 1.0],
    )
    .unwrap();
    let at = |t| {
        let s = g.snapshot(&SnapshotRequest::at(t)).unwrap();
        let c = s
            .node_block("company")
            .unwrap()
            .channel("fund")
            .unwrap()
            .clone();
        (c.present[0], c.values[0])
    };
    assert_eq!(at(99), (0, 0.0));
    assert_eq!(at(150), (1, 10.0));
    assert_eq!(at(250), (1, 8.0));
}

#[test]
fn delisting_removes_node_and_its_edges() {
    let mut g = graph();
    g.upsert_nodes(
        "company",
        &["aapl", "tsm"],
        &[0, 0],
        &[i64::MAX, i64::MAX],
        &[0, 0],
    )
    .unwrap();
    g.assert_edges(
        "supplies",
        &["tsm"],
        &["aapl"],
        &[0],
        &[i64::MAX],
        &[0],
        &[0.25],
    )
    .unwrap();
    // Learned at t=60 that tsm stopped existing at t=50.
    g.retract_nodes("company", &["tsm"], &[50], &[i64::MAX], &[60])
        .unwrap();
    let edges = |known, valid| {
        let mut r = SnapshotRequest::at(known);
        r.valid_at = Some(valid);
        g.snapshot(&r)
            .unwrap()
            .edge_block("supplies")
            .unwrap()
            .src
            .len()
    };
    assert_eq!(edges(55, 55), 1, "delisting not known yet");
    assert_eq!(edges(60, 55), 0);
    assert_eq!(edges(60, 40), 1, "history before the delisting is intact");
}

#[test]
fn event_window_bounds_articles() {
    let mut g = graph();
    g.upsert_nodes(
        "article",
        &["a1", "a2"],
        &[10, 50],
        &[i64::MAX, i64::MAX],
        &[11, 51],
    )
    .unwrap();
    let keys = |t: i64, w: Option<i64>| {
        let mut r = SnapshotRequest::at(t);
        if let Some(w) = w {
            r.windows.insert("article".into(), w);
        }
        g.snapshot(&r)
            .unwrap()
            .node_block("article")
            .unwrap()
            .keys
            .clone()
    };
    assert_eq!(keys(55, None), vec!["a2"]);
    assert_eq!(keys(40, None), vec!["a1"]);
    assert_eq!(keys(55, Some(100)), vec!["a1", "a2"]);
    assert!(keys(10, None).is_empty(), "a1 known at 11, not 10");
}

#[test]
fn strict_validation() {
    let mut g = graph();
    g.upsert_nodes("article", &["a1"], &[0], &[i64::MAX], &[0])
        .unwrap();
    assert!(
        g.observe("article", "emb", &["a1"], &[1], &[1], &[0.0, 2.0, 0.0])
            .is_err(),
        "bounds"
    );
    assert!(
        g.observe("article", "emb", &["a1"], &[1], &[1], &[0.0, f32::NAN, 0.0])
            .is_err(),
        "nan"
    );
    assert!(
        g.observe("article", "emb", &["zz"], &[1], &[1], &[0.0, 0.0, 0.0])
            .is_err(),
        "unknown node"
    );
    assert!(
        g.observe("article", "emb", &["a1"], &[1], &[1], &[0.0, 0.0])
            .is_err(),
        "shape"
    );
    assert!(
        g.observe("article", "emb", &["a1"], &[i64::MAX], &[1], &[0.0; 3])
            .is_err(),
        "sentinel"
    );
    assert!(
        g.upsert_nodes("company", &["aapl"], &[5], &[5], &[0])
            .is_err(),
        "empty interval"
    );
    let mut r = SnapshotRequest::at(1);
    r.history.insert("company.nope".into(), 2);
    assert!(g.snapshot(&r).is_err(), "unknown channel in request");
}

#[test]
fn corrupt_files_are_rejected() {
    let mut g = graph();
    g.upsert_nodes("company", &["aapl"], &[0], &[i64::MAX], &[0])
        .unwrap();
    g.observe("company", "fund", &["aapl"], &[1], &[1], &[1.0, 2.0])
        .unwrap();
    let bytes = g.to_bytes().unwrap();
    assert!(TemporalGraph::from_bytes(&bytes[..bytes.len() - 1]).is_err());
    let mut flipped = bytes.clone();
    let last = flipped.len() - 3;
    flipped[last] ^= 0x40;
    assert!(TemporalGraph::from_bytes(&flipped).is_err());
    let mut wrong_version = bytes.clone();
    wrong_version[8] = 99;
    assert!(TemporalGraph::from_bytes(&wrong_version).is_err());

    let dir = std::env::temp_dir().join(format!("moonrs-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("store.moon");
    g.save(&path).unwrap();
    let h = TemporalGraph::load(&path).unwrap();
    assert_eq!(h.to_bytes().unwrap(), bytes);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn homogeneous_view_is_consistent() {
    let mut g = graph();
    g.upsert_nodes(
        "company",
        &["aapl", "tsm", "nvda"],
        &[0; 3],
        &[i64::MAX; 3],
        &[0; 3],
    )
    .unwrap();
    g.upsert_nodes("article", &["a1"], &[5], &[i64::MAX], &[5])
        .unwrap();
    g.assert_edges(
        "supplies",
        &["tsm", "tsm"],
        &["aapl", "nvda"],
        &[0, 0],
        &[i64::MAX; 2],
        &[0, 0],
        &[0.2, f32::NAN],
    )
    .unwrap();
    g.assert_edges(
        "mentions",
        &["a1"],
        &["nvda"],
        &[5],
        &[i64::MAX],
        &[5],
        &[0.5, 0.9],
    )
    .unwrap();
    let s = g.snapshot(&SnapshotRequest::at(10)).unwrap();
    let h = s.homogeneous(true);
    // company: aapl=0, nvda=1, tsm=2 ; article: a1=3
    assert_eq!(h.kind_offsets, vec![0, 3, 4]);
    assert_eq!(h.node_type, vec![0, 0, 0, 1]);
    assert_eq!(h.src.len(), 6);
    assert_eq!(h.indptr.len(), 5);
    let triples: Vec<(i64, i64, i64)> = (0..h.src.len())
        .map(|i| (h.src[i], h.dst[i], h.edge_type[i]))
        .collect();
    assert_eq!(
        triples,
        vec![
            (0, 2, 2),
            (1, 2, 2),
            (1, 3, 3),
            (2, 0, 0),
            (2, 1, 0),
            (3, 1, 1)
        ]
    );
    for t in 0..4usize {
        let (a, b) = (h.indptr[t] as usize, h.indptr[t + 1] as usize);
        assert!(h.src[a..b].iter().all(|&x| x == t as i64));
    }
    // The nullable attribute came through as 0 with its presence bit cleared.
    let sup = s.edge_block("supplies").unwrap();
    assert_eq!(sup.attrs, vec![0.2, 0.0]);
    assert_eq!(sup.attr_present, Some(vec![1, 0]));
}
