//! Loader for the gigatoken-addon C ABI from Node.js.
//!
//! The Rust crate is built as a cdylib exposing:
//!   int32_t gt_init(const char* gguf_path, void** out)          -> 0 ok / errno
//!   void    gt_free(void* handle)
//!   int32_t gt_tokenize(void* handle, const char* text,
//!                       uint32_t* out_ids, size_t out_cap, size_t* out_len)
//!   int32_t gt_send_pretokenized(void* handle, const char* host, uint16_t port,
//!                       const char* model, const char* text, uint32_t max_tokens,
//!                       char* out, size_t out_cap)
//!
//! Node's `process.dlopen` only loads native modules that export an
//! `Initialize`/`node_register_module_v1` hook; a plain C-ABI cdylib does not.
//! Two integration options are supported:
//!   A) native N-API module (preferred): build the crate with `napi-rs` so
//!      Node imports it directly (`require('./gigatoken_addon.node')`).
//!   B) FFI bridge: load the cdylib through a small C shim exposed as a Node
//!      addon, or via `koffi` (already a dependency of the dsh profile).
//!
//! This loader implements option B using the `koffi` FFI library (present in
//! the dsh profile's node_modules) so no extra compile step is required at
//! runtime. It locates the built cdylib, binds the four C functions, and
//! re-exports a promise-based tokenizer.
//!
//! @module gigatoken-addon/loader.mjs

// Resolve the built cdylib. Prefer the crate release output.
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const ROOT = dirname(fileURLToPath(import.meta.url))
const CANDIDATES = [
  join(ROOT, 'target', 'release', 'libgigatoken_addon.so'),
  join(ROOT, 'target', 'debug', 'libgigatoken_addon.so'),
  join(ROOT, '..', 'llcd-control', 'target', 'release', 'libgigatoken_addon.so'),
]

let koffi = null
try {
  koffi = await import('koffi')
} catch {
  // not installed at this path; caller must provide it
}

function findLib() {
  for (const p of CANDIDATES) {
    try {
      require?.resolve(p)
    } catch {}
    if (typeof p === 'string' && p.includes('libgigatoken_addon.so')) {
      return p
    }
  }
  return CANDIDATES[0]
}

/**
 * Open the addon and bind its C ABI with koffi.
 * @returns {Promise<object>} { tokenize, free, sendPretokenized }
 */
export async function openAddon({ libPath = findLib() } = {}) {
  if (!koffi) throw new Error('koffi not available; install koffi or build the N-API variant')
  const lib = koffi.load(libPath)

  const gt_init = lib.func('int32_t gt_init(const char *gguf_path, void **out)')
  const gt_free = lib.func('void gt_free(void *handle)')
  const gt_tokenize = lib.func('int32_t gt_tokenize(void *handle, const char *text, uint32_t *out_ids, size_t out_cap, size_t *out_len)')
  const gt_send = lib.func('int32_t gt_send_pretokenized(void *handle, const char *host, uint16_t port, const char *model, const char *text, uint32_t max_tokens, char *out, size_t out_cap)')

  const ggufPath = process.env.BONSAI_GGUF
    ?? '/run/media/naruzkurai/Win-ntfs/Ternary-Bonsai-27B-MTP-TQ2_0.gguf'
  const handlePtr = koffi.alloc('void *', null)
  const rc = gt_init(ggufPath, handlePtr)
  if (rc !== 0) throw new Error(`gt_init failed with code ${rc}`)
  const handle = koffi.decode(handlePtr).value ?? handlePtr

  return {
    tokenize(text) {
      const outCap = 262144
      const ids = koffi.alloc('uint32_t', outCap)
      const lenPtr = koffi.alloc('size_t', 0)
      const n = gt_tokenize(handle, text, ids, outCap, lenPtr)
      const len = koffi.decode(lenPtr).value
      const arr = []
      for (let i = 0; i < len && i < n; i++) arr.push(koffi.decode(ids, i))
      koffi.free(ids); koffi.free(lenPtr)
      return arr
    },
    sendPretokenized({ host, port, model, text, maxTokens = 64 }) {
      const cap = 65536
      const out = koffi.alloc('char', cap)
      const status = gt_send(handle, host, port, model, text, maxTokens, out, cap)
      return { status, text: koffi.decode(out) }
    },
    async free() {
      gt_free(handle)
    },
  }
}
