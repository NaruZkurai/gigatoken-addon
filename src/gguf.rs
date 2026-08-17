//! Minimal GGUF (v3) metadata reader.
//!
//! Only reads what the vocab extraction needs: the KV section's tokenizer
//! arrays and the scalar special-token ids. Tensor headers and the data blob
//! are skipped, so a 8.8 GiB Bonsai GGUF is parsed in a few milliseconds from
//! the file without loading the weights. This is pure std.
//!
//! Layout implemented:
//!   magic  u32   "GGUF"
//!   version u32
//!   n_tensors i64
//!   n_kv i64
//!   <n_kv> KV pairs   { name: string, type: i32, value }
//!   <n_tensors> tensor headers (skipped)
//!   data blob (skipped)
//!
//! @module gigatoken-addon/gguf

use std::collections::HashMap;
use std::io::{Read, Seek};

/// A GGUF read error.
#[derive(Debug)]
pub struct GgufError(pub String);

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gguf: {}", self.0)
    }
}

impl std::error::Error for GgufError {}

fn err<T>(msg: impl Into<String>) -> Result<T, GgufError> {
    Err(GgufError(msg.into()))
}

/// A buffered byte reader over a `Read + Seek` source.
struct Reader<R: Read + Seek> {
    r: R,
    buf: Vec<u8>,
    pos: usize,
}

impl<R: Read + Seek> Reader<R> {
    fn new(mut r: R) -> Result<Self, GgufError> {
        let mut buf = Vec::with_capacity(1 << 16);
        let mut chunk = [0u8; 65536];
        let n = r.read(&mut chunk).map_err(|e| GgufError(e.to_string()))?;
        buf.extend_from_slice(&chunk[..n]);
        Ok(Reader { r, buf, pos: 0 })
    }
    fn ensure(&mut self, n: usize) -> Result<(), GgufError> {
        while self.buf.len().saturating_sub(self.pos) < n {
            let mut chunk = [0u8; 65536];
            let got = self.r.read(&mut chunk).map_err(|e| GgufError(e.to_string()))?;
            if got == 0 {
                return err("unexpected end of file");
            }
            self.buf.extend_from_slice(&chunk[..got]);
        }
        Ok(())
    }
    fn take(&mut self, n: usize) -> Result<&[u8], GgufError> {
        self.ensure(n)?;
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, GgufError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn i32(&mut self) -> Result<i32, GgufError> {
        Ok(self.u32()? as i32)
    }
    fn u64(&mut self) -> Result<u64, GgufError> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }
    fn i64(&mut self) -> Result<i64, GgufError> {
        Ok(self.u64()? as i64)
    }
    fn f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn string(&mut self) -> Result<String, GgufError> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }
    /// Read a string value as raw bytes (no UTF-8 validation), for token texts.
    fn raw_string(&mut self) -> Result<Vec<u8>, GgufError> {
        let len = self.u64()? as usize;
        Ok(self.take(len)?.to_vec())
    }
}

/// The tokenizer-relevant subset of a GGUF model's metadata.
#[derive(Debug, Clone)]
pub struct Vocab {
    /// `tokenizer.ggml.model` ("gpt2" for Bonsai).
    pub model: String,
    /// `tokenizer.ggml.pre` ("qwen35" for Bonsai).
    pub pre: String,
    /// `tokenizer.ggml.tokens` — raw token byte strings.
    pub tokens: Vec<Vec<u8>>,
    /// `tokenizer.ggml.merges` — "left right" pairs, in rank order.
    pub merges: Vec<(Vec<u8>, Vec<u8>)>,
    /// `tokenizer.ggml.token_type` — GGUF token types (not GT enum).
    pub token_types: Vec<u32>,
    /// `tokenizer.ggml.scores` — absent when empty.
    pub scores: Vec<f32>,
    /// Special token ids.
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub pad_id: Option<u32>,
    pub unknown_id: Option<u32>,
    /// `tokenizer.ggml.add_bos_token`.
    pub add_bos: bool,
}

impl Vocab {
    /// text -> id. Empty token bytes were normalized to "[EMPTY_i]" by the
    /// parser, so every entry has a unique key.
    pub fn text_to_id(&self) -> HashMap<Vec<u8>, u32> {
        let mut map = HashMap::with_capacity(self.tokens.len());
        for (i, t) in self.tokens.iter().enumerate() {
            map.entry(t.clone()).or_insert(i as u32);
        }
        map
    }
}

/// Parse a GGUF file by reading only its metadata KV section.
pub fn parse_path(path: &str) -> Result<Vocab, GgufError> {
    let f = std::fs::File::open(path).map_err(|e| GgufError(format!("open {path}: {e}")))?;
    let mut r = Reader::new(f)?;

    let magic = r.take(4)?;
    if magic != b"GGUF" {
        return err("bad magic");
    }
    let _version = r.u32()?;
    let _n_tensors = r.i64()?;
    let n_kv = r.i64()?;

    let mut vocab = Vocab {
        model: String::new(),
        pre: String::new(),
        tokens: Vec::new(),
        merges: Vec::new(),
        token_types: Vec::new(),
        scores: Vec::new(),
        bos_id: None,
        eos_id: None,
        pad_id: None,
        unknown_id: None,
        add_bos: false,
    };

    for _ in 0..n_kv {
        let key = r.string()?;
        let ty = r.i32()?; // gguf_type
        match key.as_str() {
            "tokenizer.ggml.model" => vocab.model = r.string()?,
            "tokenizer.ggml.pre" => vocab.pre = r.string()?,
            "tokenizer.ggml.add_bos_token" => vocab.add_bos = r.u8()? != 0,
            "tokenizer.ggml.bos_token_id" => vocab.bos_id = Some(r.u32()?),
            "tokenizer.ggml.eos_token_id" => vocab.eos_id = Some(r.u32()?),
            "tokenizer.ggml.padding_token_id" => vocab.pad_id = Some(r.u32()?),
            "tokenizer.ggml.unknown_token_id" => vocab.unknown_id = Some(r.u32()?),
            "tokenizer.ggml.tokens" => {
                let elem = r.i32()?;
                if elem != 8 {
                    return err("tokenizer.ggml.tokens is not an array of strings");
                }
                let n = r.u64()? as usize;
                let mut toks = Vec::with_capacity(n);
                for _ in 0..n {
                    toks.push(r.raw_string()?);
                }
                vocab.tokens = toks;
            }
            "tokenizer.ggml.token_type" => {
                let _ = r.i32()?; // elem type
                let n = r.u64()? as usize;
                let mut tt = Vec::with_capacity(n);
                for _ in 0..n {
                    tt.push(r.u32()?);
                }
                vocab.token_types = tt;
            }
            "tokenizer.ggml.scores" => {
                let _ = r.i32()?; // elem type
                let n = r.u64()? as usize;
                let mut sc = Vec::with_capacity(n);
                for _ in 0..n {
                    sc.push(r.f32()?);
                }
                vocab.scores = sc;
            }
            "tokenizer.ggml.merges" => {
                let elem = r.i32()?;
                if elem != 8 {
                    return err("tokenizer.ggml.merges is not an array of strings");
                }
                let n = r.u64()? as usize;
                let mut merges = Vec::with_capacity(n);
                for _ in 0..n {
                    let s = r.raw_string()?;
                    let p = s.iter().position(|&b| b == b' ').filter(|&i| i >= 1);
                    match p {
                        Some(pos) => {
                            let left = s[0..pos].to_vec();
                            let right = s[pos + 1..].to_vec();
                            merges.push((left, right));
                        }
                        None => merges.push((s, Vec::new())),
                    }
                }
                vocab.merges = merges;
            }
            _ => skip_gguf_value(&mut r, ty)?,
        }
    }

    for (i, t) in vocab.tokens.iter_mut().enumerate() {
        if t.is_empty() {
            *t = format!("[EMPTY_{i}]").into_bytes();
        }
    }
    Ok(vocab)
}

/// Skip a single GGUF value of type `ty` (gguf_type int).
fn skip_gguf_value<R: Read + Seek>(r: &mut Reader<R>, ty: i32) -> Result<(), GgufError> {
    match ty {
        0 | 1 | 7 => { r.take(1)?; }   // u8 i8 bool
        2 | 3 => { r.take(2)?; }       // u16 i16
        4 | 5 | 6 => { r.take(4)?; }   // u32 i32 f32
        8 => { r.string()?; }          // string
        9 => {                         // array
            let elem = r.i32()?;
            let n = r.u64()? as usize;
            let stride = match elem {
                0 | 1 | 7 => 1usize,
                2 | 3 => 2,
                4 | 5 | 6 => 4,
                10 | 11 | 12 => 8,
                _ => 4,
            };
            for _ in 0..n {
                if elem == 8 {
                    r.string()?;
                } else {
                    r.take(stride)?;
                }
            }
        }
        10 | 11 | 12 => { r.take(8)?; } // u64 i64 f64
        _ => return err("unknown gguf type"),
    }
    Ok(())
}
