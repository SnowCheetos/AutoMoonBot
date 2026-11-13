//! Durable, checksummed, versioned store files.
//!
//! Layout (little endian):
//!
//! ```text
//! magic        8 bytes  b"MOONRS\0\x02"
//! version      u32      FORMAT_VERSION
//! payload_len  u64
//! crc32        u32      of payload
//! payload      postcard-encoded store image
//! ```
//!
//! Writes go to a temporary file in the destination directory, are fsynced,
//! then atomically renamed into place, so a crash never leaves a torn file.
//! Loads verify the checksum and then re-validate every structural invariant
//! and every stored value against the schema, so a corrupt or hand-edited
//! file is rejected rather than producing a silently wrong snapshot.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::schema::{KindId, NodeKindSpec, RelationSpec, Schema};
use crate::store::{KindStore, RelationStore, TemporalGraph};

const MAGIC: &[u8; 8] = b"MOONRS\0\x02";
pub const FORMAT_VERSION: u32 = 1;
const HEADER: usize = 8 + 4 + 8 + 4;

#[derive(Serialize)]
struct ImageRef<'a> {
    kinds: &'a [NodeKindSpec],
    relations: &'a [RelationSpec],
    kind_stores: &'a [KindStore],
    relation_stores: &'a [RelationStore],
}

#[derive(Deserialize)]
struct Image {
    kinds: Vec<NodeKindSpec>,
    relations: Vec<RelationSpec>,
    kind_stores: Vec<KindStore>,
    relation_stores: Vec<RelationStore>,
}

impl TemporalGraph {
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let (kinds, relations) = self.schema.to_parts();
        let image = ImageRef {
            kinds: &kinds,
            relations: &relations,
            kind_stores: &self.kinds,
            relation_stores: &self.relations,
        };
        let payload = postcard::to_stdvec(&image).map_err(|e| Error::Format(e.to_string()))?;
        let mut out = Vec::with_capacity(HEADER + payload.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        out.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
        out.extend_from_slice(&payload);
        Ok(out)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER || &bytes[..8] != MAGIC {
            return Err(Error::Format("not a moonrs store file".into()));
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes"));
        if version != FORMAT_VERSION {
            return Err(Error::Format(format!(
                "format version {version} is not supported (expected {FORMAT_VERSION})"
            )));
        }
        let len = u64::from_le_bytes(bytes[12..20].try_into().expect("8 bytes"));
        let crc = u32::from_le_bytes(bytes[20..24].try_into().expect("4 bytes"));
        let payload = &bytes[HEADER..];
        if payload.len() as u64 != len {
            return Err(Error::Format("truncated payload".into()));
        }
        if crc32fast::hash(payload) != crc {
            return Err(Error::Format("checksum mismatch".into()));
        }
        let image: Image =
            postcard::from_bytes(payload).map_err(|e| Error::Format(e.to_string()))?;
        Self::from_image(image)
    }

    fn from_image(image: Image) -> Result<Self> {
        let schema = Schema::rebuild(image.kinds, image.relations)
            .map_err(|e| Error::Format(format!("invalid schema: {e}")))?;
        if image.kind_stores.len() != schema.kinds().len()
            || image.relation_stores.len() != schema.relations().len()
        {
            return Err(Error::Format("store/schema length mismatch".into()));
        }
        let mut graph = TemporalGraph {
            schema,
            kinds: image.kind_stores,
            relations: image.relation_stores,
        };
        graph.validate_loaded()?;
        graph.rebuild_indexes()?;
        Ok(graph)
    }

    fn validate_loaded(&mut self) -> Result<()> {
        let bad = |what: String| Err(Error::Format(what));
        for (ki, store) in self.kinds.iter_mut().enumerate() {
            let spec = self.schema.kind(KindId(ki as u16));
            let n = store.keys.len();
            if store.lifetimes.len() != n || store.channels.len() != spec.channels.len() {
                return bad(format!("{}: inconsistent node tables", spec.name));
            }
            if !store.lifetimes.iter().all(|l| l.check()) {
                return bad(format!(
                    "{}: lifetime versions violate invariants",
                    spec.name
                ));
            }
            for (ci, series_vec) in store.channels.iter_mut().enumerate() {
                let cspec = &spec.channels[ci];
                if series_vec.len() > n {
                    return bad(format!(
                        "{}.{}: series for missing nodes",
                        spec.name, cspec.name
                    ));
                }
                for s in series_vec.iter_mut() {
                    if !s.finish_load(cspec.value.dim) {
                        return bad(format!("{}.{}: corrupt series", spec.name, cspec.name));
                    }
                    for r in 0..s.len() {
                        cspec
                            .value
                            .validate(s.row(r, cspec.value.dim), || {
                                format!("{}.{}", spec.name, cspec.name)
                            })
                            .map_err(|e| Error::Format(e.to_string()))?;
                    }
                }
            }
        }
        for (ri, store) in self.relations.iter().enumerate() {
            let spec = &self.schema.relations()[ri];
            let (ns, nd) = (
                self.kinds[spec.src.0 as usize].keys.len(),
                self.kinds[spec.dst.0 as usize].keys.len(),
            );
            for (&(s, d), list) in &store.edges {
                if s as usize >= ns || d as usize >= nd || list.is_empty() || !list.check() {
                    return bad(format!("{}: corrupt edge table", spec.name));
                }
                for v in list.versions() {
                    if let Some(a) = &v.payload {
                        spec.attrs
                            .validate(a, || spec.name.clone())
                            .map_err(|e| Error::Format(e.to_string()))?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Atomically write the store to `path`.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let bytes = self.to_bytes()?;
        let dir = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let file_name = path
            .file_name()
            .ok_or_else(|| Error::Io(format!("{} is not a file path", path.display())))?;
        let tmp = dir.join(format!(
            ".{}.tmp-{}",
            file_name.to_string_lossy(),
            std::process::id()
        ));
        let result = (|| {
            let mut f = File::create(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
            drop(f);
            fs::rename(&tmp, path)?;
            // Persist the rename itself.
            #[cfg(unix)]
            File::open(dir)?.sync_all()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&tmp);
        }
        result
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let mut bytes = Vec::new();
        File::open(path)?.read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes)
    }
}
