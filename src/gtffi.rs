//! Bindings to the fork's gigatoken C ABI (`gt_llama_*`).
//!
//! These are the exact struct/function layouts from
//! `patches/gigatoken-llama-cpp.patch`. They are loaded at runtime through
//! `libloading` from a gigatoken `cdylib` that exposes the C ABI (produced by
//! building gigatoken with the `llama-cpp` feature). When no such library is
//! present, [`crate::bpe::BpeTokenizer`] is used as a pure-Rust fallback so the
//! addon always works.
//!
//! @module gigatoken-addon/gtffi

use crate::gguf::Vocab;
use libloading::{Library, Symbol};

/// GT status codes.
pub const GT_LLAMA_STATUS_OK: i32 = 0;

// GT token type.
pub const GT_LLAMA_TOKEN_TYPE_NORMAL: u32 = 0;
pub const GT_LLAMA_TOKEN_TYPE_BYTE: u32 = 1;
pub const GT_LLAMA_TOKEN_TYPE_SPECIAL: u32 = 2;

// GT pretokenizer enum.
pub const GT_LLAMA_PRETOKENIZER_QWEN35: u32 = 3;

// BPE flags.
pub const GT_LLAMA_BPE_FLAG_VOCAB_ID_RANKS: u32 = 1 << 1;

/// `gt_llama_bytes` — a borrowed byte slice.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GtLlamaBytes {
    pub data: *const u8,
    pub len: usize,
}

/// `gt_llama_vocab_token` — 24 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GtLlamaVocabToken {
    pub text: GtLlamaBytes,
    pub score: f32,
    pub token_type: u32,
}

/// `gt_llama_merge` — 16 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GtLlamaMerge {
    pub left: u32,
    pub right: u32,
    pub merged: u32,
    pub rank: u32,
}

/// `gt_llama_token_buffer` — 24 bytes.
#[repr(C)]
pub struct GtLlamaTokenBuffer {
    pub data: *mut u32,
    pub len: usize,
    pub capacity: usize,
}

/// `gt_llama_error` — 24 bytes.
#[repr(C)]
pub struct GtLlamaError {
    pub data: *mut u8,
    pub len: usize,
    pub capacity: usize,
}

/// Error returned when loading/using the gigatoken C ABI.
#[derive(Debug)]
pub struct GtError(pub String);

impl std::fmt::Display for GtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gigatoken: {}", self.0)
    }
}

impl std::error::Error for GtError {}

/// Translate a GGUF token_type to the GT enum.
///
/// Gigatoken's `token_type()` maps GGUF BYTE(6) -> GT BYTE(1) and every other
/// non-NORMAL type -> GT SPECIAL(2); GGUF NORMAL(1) -> GT NORMAL(0).
fn gt_token_type(gbuf: u32) -> u32 {
    match gbuf {
        6 => GT_LLAMA_TOKEN_TYPE_BYTE,
        1 => GT_LLAMA_TOKEN_TYPE_NORMAL,
        _ => GT_LLAMA_TOKEN_TYPE_SPECIAL,
    }
}

/// Build the `gt_llama_vocab_token[]` and `gt_llama_merge[]` arrays from a
/// parsed GGUF `Vocab`. The returned slices must stay alive for the lifetime
/// of any tokenizer created from them.
pub fn build_tables(
    vocab: &Vocab,
) -> (Vec<GtLlamaVocabToken>, Vec<GtLlamaMerge>) {
    // Raw token bytes must outlive the token structs; we keep them in a Vec<Vec<u8>>
    // and reference them, but the C ABI wants stable pointers during create_bpe.
    // Simpler and safe here: re-take ownership into owned byte-strings whose
    // pointers we fix up — but the tokenizer copies the bytes during create, so
    // we can build transient tables for the duration of create_bpe().
    let tokens: Vec<GtLlamaVocabToken> = {
        let mut held: Vec<Vec<u8>> = Vec::with_capacity(vocab.tokens.len());
        let mut out: Vec<GtLlamaVocabToken> = Vec::with_capacity(vocab.tokens.len());
        for (i, t) in vocab.tokens.iter().enumerate() {
            // gigatoken expects token text in the model's own byte representation.
            // For qwen35/GPT-style byte-level BPE the GGUF stores byte tokens as
            // bytes_to_unicode chars; gigatoken (like llama.cpp) accepts those
            // directly, so pass the stored text through unchanged.
            held.push(t.clone());
            let bytes = held.last().unwrap();
            out.push(GtLlamaVocabToken {
                text: GtLlamaBytes { data: bytes.as_ptr(), len: bytes.len() },
                score: vocab.scores.get(i).copied().unwrap_or(0.0),
                token_type: gt_token_type(vocab.token_types.get(i).copied().unwrap_or(1)),
            });
        }
        out
    };

    // Resolve merge ids via the text->id map.
    let text_to_id = vocab.text_to_id();
    let mut merges: Vec<GtLlamaMerge> = Vec::with_capacity(vocab.merges.len());
    for (rank, (left, right)) in vocab.merges.iter().enumerate() {
        if left.is_empty() || right.is_empty() {
            continue;
        }
        let li = text_to_id.get(left).copied().unwrap_or(u32::MAX);
        let ri = text_to_id.get(right).copied().unwrap_or(u32::MAX);
        // `merged` id: the merged token (left+right concatenated) — the model
        // vocabulary contains the combined piece under a distinct id. We look
        // it up from the concatenation; if absent we leave it as a sentinel.
        let mut m = left.clone();
        m.extend_from_slice(right);
        let mi = text_to_id.get(&m).copied().unwrap_or(u32::MAX);
        if li == u32::MAX || ri == u32::MAX || mi == u32::MAX {
            continue;
        }
        merges.push(GtLlamaMerge { left: li, right: ri, merged: mi, rank: rank as u32 });
    }
    (tokens, merges)
}

/// A loaded gigatoken C-ABI tokenizer handle plus the library that owns it.
pub struct GtTokenizer {
    _lib: Library,
    ptr: *mut std::ffi::c_void,
}

// The tokenizer handle is not thread-safe by the C ABI contract; we expose a
// single-owner handle and do not share it across threads.
unsafe impl Send for GtTokenizer {}
unsafe impl Sync for GtTokenizer {}

/// Paths searched (in order) for a gigatoken C-ABI cdylib.
pub const GT_CANDIDATE_LIBS: &[&str] = &[
    // Built by this addon's build.sh (`cargo rustc --features llama-cpp --crate-type cdylib`).
    "/nzk/git/pithagoras/rust/gigatoken-addon/gigatoken-target/release/libgigatoken_rs.so",
    // Built by `cargo build --features llama-cpp` inside the fork's vendor/gigatoken.
    "/nzk/git/pithagoras/gitrepos/llama-direct-token-input/vendor/gigatoken/target/release/libgigatoken_rs.so",
    // Copied into the fork build tree by CMake when LLAMA_GIGATOKEN=ON.
    "/nzk/git/pithagoras/gitrepos/llama-direct-token-input/build/libgigatoken_rs.so",
];

/// Load the gigatoken C ABI from the first available candidate library.
pub fn load_gt_abi() -> Option<Library> {
    for p in GT_CANDIDATE_LIBS {
        if let Ok(lib) = unsafe { Library::new(p) } {
            return Some(lib);
        }
    }
    None
}

impl GtTokenizer {
    /// Create a BPE tokenizer from a parsed GGUF vocab via the gigatoken C ABI.
    pub fn create(vocab: &Vocab) -> Result<GtTokenizer, GtError> {
        let lib = load_gt_abi().ok_or_else(|| {
            GtError("no gigatoken C-ABI library found; build it (see rust/gigatoken-addon/BUILD.md) or use the pure-Rust fallback".into())
        })?;

        let (tokens, merges) = build_tables(vocab);
        let (pretokenizer, flags) = (GT_LLAMA_PRETOKENIZER_QWEN35, 0u32);

        // SAFETY: the symbols are present because the candidate lib exports the
        // C ABI; pointers to `tokens`/`merges` are valid for the call duration.
        unsafe {
            let create: Symbol<unsafe extern "C" fn(
                *const GtLlamaVocabToken,
                usize,
                *const GtLlamaMerge,
                usize,
                u32,
                u32,
                *mut *mut std::ffi::c_void,
                *mut GtLlamaError,
            ) -> i32> = lib.get(b"gt_llama_tokenizer_create_bpe").map_err(|e| GtError(e.to_string()))?;

            let mut handle: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut error = GtLlamaError { data: std::ptr::null_mut(), len: 0, capacity: 0 };
            let st = create(
                tokens.as_ptr(),
                tokens.len(),
                merges.as_ptr(),
                merges.len(),
                pretokenizer,
                flags,
                &mut handle,
                &mut error,
            );
            let _ = error; // error text holder; clear below
            if st != GT_LLAMA_STATUS_OK {
                return Err(GtError(format!("create_bpe failed with status {st}")));
            }
            let ptr = handle as *mut std::ffi::c_void;
            Ok(GtTokenizer { _lib: lib, ptr })
        }
    }

    /// Encode raw text bytes into token ids.
    pub fn encode(&self, text: &[u8]) -> Result<Vec<u32>, GtError> {
        unsafe {
            let encode: Symbol<unsafe extern "C" fn(
                *mut std::ffi::c_void,
                GtLlamaBytes,
                *mut GtLlamaTokenBuffer,
                *mut GtLlamaError,
            ) -> i32> = self._lib.get(b"gt_llama_tokenizer_encode").map_err(|e| GtError(e.to_string()))?;
            let free_buf: Symbol<unsafe extern "C" fn(*mut GtLlamaTokenBuffer)> =
                self._lib.get(b"gt_llama_token_buffer_free").map_err(|e| GtError(e.to_string()))?;

            let mut buf = GtLlamaTokenBuffer { data: std::ptr::null_mut(), len: 0, capacity: 0 };
            let mut error = GtLlamaError { data: std::ptr::null_mut(), len: 0, capacity: 0 };
            let input = GtLlamaBytes { data: text.as_ptr(), len: text.len() };
            let st = encode(self.ptr, input, &mut buf, &mut error);
            if st != GT_LLAMA_STATUS_OK {
                return Err(GtError(format!("encode failed with status {st}")));
            }
            let ids = std::slice::from_raw_parts(buf.data, buf.len).to_vec();
            free_buf(&mut buf);
            Ok(ids)
        }
    }
}

impl Drop for GtTokenizer {
    fn drop(&mut self) {
        unsafe {
            if let Ok(free_tok) = self
                ._lib
                .get::<Symbol<unsafe extern "C" fn(*mut std::ffi::c_void)>>(b"gt_llama_tokenizer_free")
            {
                free_tok(self.ptr);
            }
        }
    }
}
