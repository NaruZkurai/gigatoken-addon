# gigatoken-addon

A Rust native addon that lets DeepSeek Harness (or any OpenAI-compatible client)
tokenize the model's prompt **locally** and send it to the
[`llama-direct-token-input`](../../gitrepos/llama-direct-token-input) fork
server through the fork's **direct-token API** (`/v1/chat_pretokenized`).

This is intended to speed up the harness's large agent prompts (~12k tokens) by
moving tokenization off the (slow) 27B server and into a fast local module.

## Pipeline

```
Bonsai GGUF  ── gguf.rs ──>  Vocab (tokens, merges, types)
                                   │
                                   ▼
                        Tokenizer: gigatoken C ABI   (preferred)
                                   │  └─ falls back to pure-Rust byte-level BPE
                                   ▼
                    local token ids ──────────────┐
                                                 ▼
            pretok.rs ── POST /v1/chat_pretokenized ──> fork server
```

## Layout

| File | Purpose |
| --- | --- |
| `src/gguf.rs` | Streaming GGUF v3 parser (metadata only; reads tokenizer KV). Pure std. |
| `src/gtffi.rs` | `#[repr(C)]` bindings + runtime `libloading` loader for gigatoken's `gt_llama_*` C ABI. |
| `src/bpe.rs` | Pure-Rust byte-level BPE fallback (no external deps). |
| `src/bytes_to_unicode.rs` | GPT-2 bytes_to_unicode reverse table (for byte-token transcoding). |
| `src/pretok.rs` | Pure-std HTTP client for `/v1/chat_pretokenized` and `/v1/chat/completions/input_tokens`. |
| `src/lib.rs` | High-level `Tokenizer` + the `gt_init` / `gt_tokenize` / `gt_free` / `gt_send_pretokenized` C entrypoints. |
| `src/bin/gt_cli.rs` | CLI smoke test. |
| `loader.mjs` | Node FFI loader (koffi) for the cdylib. |
| `build.sh` | Stages + patches + builds gigatoken cdylib, then builds this crate. |

## Build

```sh
./build.sh        # builds gigatoken cdylib + this crate
```

or build just this crate (pure-Rust fallback only):

```sh
cargo build --release
```

The gigatoken C-ABI cdylib is searched (in order) at:

```text
rust/gigatoken-addon/gigatoken-target/release/libgigatoken_rs.so
…/llama-direct-token-input/vendor/gigatoken/target/release/libgigatoken_rs.so
…/llama-direct-token-input/build/libgigatoken_rs.so
```

## The direct-token endpoint it speaks to

The fork server registers (see `tools/server/server.cpp`):

```text
POST /v1/chat_pretokenized            -> post_chat_completions_pretokenized
POST /v1/chat/completions/input_tokens -> post_chat_completions_tok (count)
```

A request body is a normal chat-completions body; any message `content` may be:

```json
{ "type": "text", "text": "Summarize: " },
{ "type": "input_tokens", "tokens": [9419, 11, 1814, 0] }
```

or a whole message whose `content` is a raw integer array. The server renders
the chat template and splices the raw ids into the prompt — no server-side
re-tokenization of the pre-encoded bulk.

## Status / known limitation (2026-08-17)

The scaffold is complete and builds. Verified working:

- GGUF metadata parses the real 27B MTP model (248320 tokens / 247587 merges).
- Direct-token HTTP client reaches `/v1/chat_pretokenized` and surfaces the
  server response / connection errors.
- gigatoken C ABI cdylib builds, loads, and the `gt_llama_*` symbols bind.

Not yet byte-exact (this is the remaining work):

- gigatoken's `gt_llama_tokenizer_create_bpe` returns status 3
  (`INVALID_MODEL`) for the qwen35 Bonsai vocab with the table built here;
  its merge validation reports `merge (220, 220) does not concatenate to token
  256`. gigatoken expects the byte-token `text`/merge relationship in a specific
  internal byte representation that this from-GGUF reconstruction does not yet
  match llama.cpp's in-memory `llama_vocab` (the fork's own C++ layer builds
  tables the same way but from the *loaded* `llama_vocab`, which transcodes
  byte tokens during GGUF load).
- The pure-Rust BPE fallback runs but over-merges (not byte-exact with the
  server's `/tokenize`).

Until local ids are byte-identical to `POST /tokenize` on the model server, the
direct-token prompts would be mis-tokenized. Two concrete ways forward:

1. Reuse the fork's `llama_vocab` loader (link llama.cpp / call its GGUF-to-vocab
   load) so tables are built exactly as the fork does, then hand those to
   gigatoken.
2. Mirror llama.cpp's byte-token transcoding (bytes-to-unicode → raw bytes) in
   **both** the token table and the merge table consistently, then re-validate
   against `POST /tokenize`.

## Node usage

`loader.mjs` loads the cdylib via koffi (present in the dsh profile). Set
`BONSAI_GGUF` to the model GGUF path, then:

```js
const { openAddon } = await import('./loader.mjs')
const tok = await openAddon()
console.log(tok.tokenize('Hello!'))             // token ids
const r = await tok.sendPretokenized({ host: '192.168.2.64', port: 6464,
  model: '/nzk/models/Ternary-Bonsai-27B-MTP-TQ2_0.gguf', text: 'hi' })
```
