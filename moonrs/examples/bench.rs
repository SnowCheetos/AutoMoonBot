//! Throughput benchmark on a synthetic, realistically shaped universe.
//!
//! `cargo run --release --example bench`
//!
//! 3,000 securities × 10 years of daily bars, quarterly fundamentals with
//! restatements, 30k supply-chain edges, 200k articles with 64-d embeddings
//! and 3 mentions each. Reports ingestion throughput and snapshot latency.

use std::time::Instant;

use moonrs::{ChannelSpec, NodeClass, NodeKindSpec, SnapshotRequest, TemporalGraph, ValueSpec};

const DAY: i64 = 86_400 * 1_000_000_000;
const N_SEC: usize = 3_000;
const N_DAYS: i64 = 2_520;
const N_ART: usize = 200_000;
const EMB: usize = 64;

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 33
}

fn main() -> moonrs::Result<()> {
    let mut g = TemporalGraph::new();
    let ch = |name: &str, dim, nullable| ChannelSpec {
        name: name.into(),
        value: ValueSpec::new(dim, nullable, None).unwrap(),
    };
    g.define_kind(NodeKindSpec {
        name: "company".into(),
        class: NodeClass::Entity,
        channels: vec![ch("bar", 5, false), ch("fundamentals", 24, true)],
    })?;
    g.define_kind(NodeKindSpec {
        name: "article".into(),
        class: NodeClass::Event { window_ns: 3 * DAY },
        channels: vec![ch("embedding", EMB, false)],
    })?;
    g.define_relation(
        "supplies",
        "company",
        "company",
        ValueSpec::new(1, false, None)?,
    )?;
    g.define_relation(
        "mentions",
        "article",
        "company",
        ValueSpec::new(2, false, None)?,
    )?;

    let keys: Vec<String> = (0..N_SEC).map(|i| format!("sec{i:05}")).collect();
    g.upsert_nodes(
        "company",
        &keys,
        &vec![0; N_SEC],
        &vec![i64::MAX; N_SEC],
        &vec![0; N_SEC],
    )?;

    // Daily bars, one batch per day (how a live pipeline would ingest).
    let start = Instant::now();
    let mut bar = vec![0f32; N_SEC * 5];
    let mut s = 1u64;
    for day in 1..=N_DAYS {
        for x in bar.iter_mut() {
            *x = (lcg(&mut s) % 10_000) as f32 / 100.0;
        }
        let valid = vec![day * DAY; N_SEC];
        let known = vec![day * DAY + 900_000_000_000; N_SEC];
        g.observe("company", "bar", &keys, &valid, &known, &bar)?;
    }
    let n_bars = N_SEC as f64 * N_DAYS as f64;
    let dt = start.elapsed().as_secs_f64();
    println!(
        "bars:         {:>10.0} rows in {dt:.2}s  ({:.1} M rows/s)",
        n_bars,
        n_bars / dt / 1e6
    );

    // Quarterly fundamentals reported ~40 days after quarter end; 10% restated.
    let start = Instant::now();
    let mut rows = 0usize;
    let fund = vec![1f32; N_SEC * 24];
    for q in 1..=(N_DAYS / 63) {
        let valid = vec![q * 63 * DAY; N_SEC];
        let known = vec![(q * 63 + 40) * DAY; N_SEC];
        g.observe("company", "fundamentals", &keys, &valid, &known, &fund)?;
        let restated: Vec<String> = keys.iter().step_by(10).cloned().collect();
        let m = restated.len();
        g.observe(
            "company",
            "fundamentals",
            &restated,
            &vec![q * 63 * DAY; m],
            &vec![(q * 63 + 120) * DAY; m],
            &vec![0.9f32; m * 24],
        )?;
        rows += N_SEC + m;
    }
    println!(
        "fundamentals: {rows:>10} rows in {:.2}s",
        start.elapsed().as_secs_f64()
    );

    // Supply chain: 10 suppliers per company.
    let start = Instant::now();
    let (mut src, mut dst) = (vec![], vec![]);
    for i in 0..N_SEC {
        for _ in 0..10 {
            src.push(keys[(lcg(&mut s) as usize) % N_SEC].clone());
            dst.push(keys[i].clone());
        }
    }
    let (src, dst): (Vec<_>, Vec<_>) = {
        let mut pairs: Vec<_> = src.into_iter().zip(dst).filter(|(a, b)| a != b).collect();
        pairs.sort();
        pairs.dedup();
        pairs.into_iter().unzip()
    };
    let m = src.len();
    g.assert_edges(
        "supplies",
        &src,
        &dst,
        &vec![0; m],
        &vec![i64::MAX; m],
        &vec![0; m],
        &vec![0.1; m],
    )?;
    println!(
        "supply edges: {m:>10} in {:.2}s",
        start.elapsed().as_secs_f64()
    );

    // Articles spread over the period, 3 mentions each.
    let start = Instant::now();
    let akeys: Vec<String> = (0..N_ART).map(|i| format!("art{i:07}")).collect();
    let at: Vec<i64> = (0..N_ART)
        .map(|i| i as i64 * (N_DAYS * DAY / N_ART as i64) + DAY)
        .collect();
    g.upsert_nodes("article", &akeys, &at, &vec![i64::MAX; N_ART], &at)?;
    let emb: Vec<f32> = (0..N_ART * EMB).map(|i| (i % 7) as f32 * 0.1).collect();
    g.observe("article", "embedding", &akeys, &at, &at, &emb)?;
    let (mut ms, mut md, mut mt) = (vec![], vec![], vec![]);
    for (i, k) in akeys.iter().enumerate() {
        for j in 0..3 {
            ms.push(k.clone());
            md.push(keys[(i * 7 + j * 13) % N_SEC].clone());
            mt.push(at[i]);
        }
    }
    let m = ms.len();
    g.assert_edges(
        "mentions",
        &ms,
        &md,
        &mt,
        &vec![i64::MAX; m],
        &mt,
        &vec![0.5; m * 2],
    )?;
    println!(
        "articles:     {N_ART:>10} (+{m} mentions) in {:.2}s",
        start.elapsed().as_secs_f64()
    );

    // Snapshot latency.
    for (label, hist) in [("K=1", 1usize), ("K=64 bars", 64)] {
        let mut req = SnapshotRequest::at(2_000 * DAY);
        req.history.insert("company.bar".into(), hist);
        req.history
            .insert("company.fundamentals".into(), hist.min(8));
        let _ = g.snapshot(&req)?;
        let iters = 20;
        let start = Instant::now();
        let mut snap = None;
        for i in 0..iters {
            req.known_at = (1_500 + i) * DAY;
            snap = Some(g.snapshot(&req)?);
        }
        let ms = start.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let snap = snap.unwrap();
        let start = Instant::now();
        let h = snap.homogeneous(true);
        let hms = start.elapsed().as_secs_f64() * 1e3;
        println!(
            "snapshot {label:<10} {ms:>7.2} ms  ({} articles, {} edges; homogeneous view {hms:.2} ms, {} tokens)",
            snap.node_block("article").unwrap().keys.len(),
            snap.edges.iter().map(|e| e.src.len()).sum::<usize>(),
            h.node_type.len(),
        );
    }
    let start = Instant::now();
    let bytes = g.to_bytes()?;
    let ser = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let _ = TemporalGraph::from_bytes(&bytes)?;
    println!(
        "persist: {:.0} MB, save {ser:.2}s, load+validate {:.2}s",
        bytes.len() as f64 / 1e6,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
