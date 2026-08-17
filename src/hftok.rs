//! Local tokenizer built from the model server's `tokenizer.json`.
//!
//! This is the "use a personal tokenizer" half of the flow:
//!   1. [`pretok::fetch_tokenizer_json`] pulls the server's HuggingFace
//!      tokenizer.json (`GET /v1/chat_tokenizer`) — the byte-exact vocab for
//!      the loaded model.
//!   2. [`LocalHfTokenizer::from_json`] loads it with gigatoken's
//!      `load_hf_slice` (differential-tested to match llama.cpp), after
//!      normalizing the qwen35 pre_tokenizer regex to gigatoken's canonical
//!      spelling.
//!   3. [`LocalHfTokenizer::encode`] tokenizes the prompt locally, producing
//!      ids identical to what the model server would.
//!
//! The resulting ids are then shipped to `/v1/chat_pretokenized`, so the
//! server never runs the tokenizer over the prompt — saving its compute.
//!
//! @module gigatoken-addon/hftok

use crate::pretok;
use gigatoken_rs::Tokenizer;
use gigatoken_rs::load_tokenizer::hf::{HfTokenizer, load_hf_slice};

/// The server/jinja spelling of the qwen35 regex: apostrophes written out as
/// explicit case classes instead of an inline `(?i:...)` group.
const SERVER_QWEN35_PREFIX: &str =
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])";

/// Normalize the pre_tokenizer `Split` regex in a fetched tokenizer.json so
/// gigatoken's exact-string recognizer maps it to the Qwen35 fast
/// pretokenizer. The server's regex is equivalent; we only fix the spelling.
fn normalize_pre_tokenizer(json: &[u8]) -> Vec<u8> {
    let s = String::from_utf8_lossy(json);
    let replaced = s.replace(SERVER_QWEN35_PREFIX, "(?i:'s|'t|'re|'ve|'m|'ll|'d)");
    replaced.into_bytes()
}

/// A locally-loaded byte-level BPE tokenizer from a server `tokenizer.json`.
///
/// The Bonsai model is byte-level BPE (`model.type="BPE"`,
/// `byte_fallback=false`), which gigatoken loads as a tiktoken-style
/// [`Tokenizer`]. SentencePiece-backed files are rejected with a clear error.
pub struct LocalHfTokenizer {
    tok: Tokenizer,
}

impl LocalHfTokenizer {
    /// Load from the raw bytes of a HuggingFace `tokenizer.json`.
    pub fn from_json(data: &[u8]) -> Result<LocalHfTokenizer, String> {
        let normalized = normalize_pre_tokenizer(data);
        let hf = load_hf_slice(&normalized).map_err(|e| format!("gigatoken load_hf_slice: {e}"))?;
        let tok = match hf {
            HfTokenizer::Bpe(t) => t,
            HfTokenizer::SentencePiece(_) => {
                return Err("tokenizer.json is SentencePiece-backed; this model is byte-level BPE".into());
            }
        };
        Ok(LocalHfTokenizer { tok })
    }

    /// Pull the tokenizer.json from a model server and load it.
    pub fn from_server(host: &str, port: u16, timeout_ms: u64) -> Result<LocalHfTokenizer, String> {
        let json = pretok::fetch_tokenizer_json(host, port, timeout_ms)?;
        Self::from_json(&json)
    }

    /// Encode raw text into token ids, byte-exact with the model server.
    pub fn encode(&mut self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        self.tok.encode_with_added_tokens_flat(text.as_bytes(), &mut out);
        out
    }
}
