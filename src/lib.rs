//! gigatoken-addon
//!
//! A local tokenizer addon for DeepSeek Harness backed by the
//! `llama-direct-token-input` fork's direct-token API.
//!
//! Pipeline:
//!   1. Parse the Bonsai GGUF metadata to recover the BPE vocab.
//!   2. Tokenize locally — via the gigatoken C ABI when its cdylib is built
//!      (the differential-tested tokenizer the fork ships), otherwise via the
//!      pure-Rust byte-level BPE fallback.
//!   3. Send pre-tokenized prompts to the fork server's `/v1/chat_pretokenized`
//!      using the `{"type":"input_tokens","tokens":[...]}` parts form, so the
//!      server skips re-tokenizing the bulk prompt.
//!
//! The crate is a `cdylib`; it exports the C functions below so a host (Node
//! via a small FFI loader, or another process) can call `gt_init` /
//! `gt_tokenize` / `gt_send_pretokenized` / `gt_free`.
//!
//! @module gigatoken-addon

pub mod bpe;
pub mod bytes_to_unicode;
pub mod gguf;
pub mod gtffi;
pub mod pretok;

use std::ffi::CStr;
use std::os::raw::c_char;

/// A tokenizer ready to use.
pub enum Tokenizer {
    /// gigatoken C ABI (loaded native lib).
    Gigatoken(gtffi::GtTokenizer),
    /// pure-Rust fallback.
    Bpe(bpe::BpeTokenizer),
}

impl Tokenizer {
    /// Build from a parsed vocab, preferring gigatoken.
    pub fn from_vocab(vocab: &gguf::Vocab) -> Result<Tokenizer, gtffi::GtError> {
        match gtffi::GtTokenizer::create(vocab) {
            Ok(t) => Ok(Tokenizer::Gigatoken(t)),
            Err(_) => Ok(Tokenizer::Bpe(bpe::BpeTokenizer::new(vocab))),
        }
    }

    /// Encode raw bytes into token ids.
    pub fn encode(&self, text: &[u8]) -> Vec<u32> {
        match self {
            Tokenizer::Gigatoken(t) => t.encode(text).unwrap_or_default(),
            Tokenizer::Bpe(t) => t.encode(text),
        }
    }
}

// ---------------------------------------------------------------------------
// C ABI exposed to the host (Node FFI / dylib loader)
// ---------------------------------------------------------------------------

/// Parse a GGUF file and build a tokenizer; stores an owned handle at `*out`.
/// Returns 0 on success, nonzero (errno-style) on failure.
/// # Safety
/// `path` must be a valid NUL-terminated C string; `out` a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn gt_init(path: *const c_char, out: *mut *mut std::ffi::c_void) -> i32 {
    if path.is_null() || out.is_null() {
        return 2; // INVALID_ARGUMENT
    }
    let path = match CStr::from_ptr(path).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return 2,
    };
    let vocab = match gguf::parse_path(&path) {
        Ok(v) => v,
        Err(_) => return 3, // INVALID_MODEL
    };
    let tok = match Tokenizer::from_vocab(&vocab) {
        Ok(t) => t,
        Err(_) => return 3,
    };
    let boxed = Box::new(tok);
    *out = Box::into_raw(boxed) as *mut std::ffi::c_void;
    0
}

/// Free a tokenizer created by `gt_init`.
/// # Safety
/// `handle` must be a pointer from `gt_init` (or NULL).
#[no_mangle]
pub unsafe extern "C" fn gt_free(handle: *mut std::ffi::c_void) {
    if handle.is_null() {
        return;
    }
    drop(Box::from_raw(handle as *mut Tokenizer));
}

/// Encode a NUL-terminated UTF-8 text into token ids.
/// Writes the ids into `out_ids` (buffer size `out_cap`) and sets `*out_len`.
/// Returns the number of ids written, or a negative errno.
/// # Safety
/// All pointers must be valid for their described sizes; text NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn gt_tokenize(
    handle: *mut std::ffi::c_void,
    text: *const c_char,
    out_ids: *mut u32,
    out_cap: usize,
    out_len: *mut usize,
) -> i32 {
    if handle.is_null() || text.is_null() || out_ids.is_null() || out_len.is_null() {
        return -2;
    }
    let tok = &*(handle as *const Tokenizer);
    let s = CStr::from_ptr(text).to_bytes();
    let ids = tok.encode(s);
    let n = ids.len().min(out_cap);
    std::ptr::copy_nonoverlapping(ids.as_ptr(), out_ids, n);
    *out_len = ids.len();
    ids.len() as i32
}

/// Send a single user message (text) to the server via `/v1/chat_pretokenized`,
/// after tokenizing `text` locally into raw ids.
/// Returns the HTTP status (200 ok) or a negative error.
/// # Safety
/// All pointers are valid NUL-terminated C strings / valid buffers.
#[no_mangle]
pub unsafe extern "C" fn gt_send_pretokenized(
    handle: *mut std::ffi::c_void,
    host: *const c_char,
    port: u16,
    model: *const c_char,
    text: *const c_char,
    max_tokens: u32,
    out: *mut c_char,
    out_cap: usize,
) -> i32 {
    if handle.is_null() || host.is_null() || model.is_null() || text.is_null() || out.is_null() {
        return -2;
    }
    let tok = &*(handle as *const Tokenizer);
    let host = CStr::from_ptr(host).to_str().unwrap_or("").to_string();
    let model = CStr::from_ptr(model).to_str().unwrap_or("").to_string();
    let text = CStr::from_ptr(text).to_str().unwrap_or("").to_string();
    let ids = tok.encode(text.as_bytes());

    let msg = pretok::Msg {
        role: "user".to_string(),
        parts: vec![pretok::Part::Text(text), pretok::Part::InputTokens(ids)],
    };
    match pretok::chat_pretokenized(&host, port, &model, &[msg], max_tokens, 0.0, false, 120_000) {
        Ok(resp) => {
            write_into(resp.body, out, out_cap);
            resp.status as i32
        }
        Err(e) => {
            write_into(e, out, out_cap);
            -1
        }
    }
}

/// Copy a Rust string into a fixed C buffer (NUL-terminated), truncated to fit.
fn write_into(s: String, out: *mut c_char, out_cap: usize) {
    let bytes = s.as_bytes();
    let n = bytes.len().min(out_cap.saturating_sub(1));
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out as *mut u8, n);
        *out.add(n) = 0;
    }
}
