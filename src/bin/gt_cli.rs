//! CLI smoke test: tokenize via the model server's own tokenizer
//! (`POST /tokenize`) and send the resulting ids through the direct-token
//! endpoint (`POST /v1/chat_pretokenized`).
//!
//! Usage:
//!   gt_cli [gguf] <text> <host> <port> [model]
//!
//! When a host/port is given, tokenization is sourced from the server's
//! `/tokenize` (byte-exact for that server's model); the tokens are then sent
//! as the pretokenized array. The local GGUF parse is only printed as a
//! diagnostic / fallback when no server is supplied.
use gigatoken_addon::{gguf, pretok, Tokenizer};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let gguf_path = args.get(1).cloned().unwrap_or_else(|| {
        "/run/media/naruzkurai/Win-ntfs/Ternary-Bonsai-27B-MTP-TQ2_0.gguf".to_string()
    });
    let text = args.get(2).cloned().unwrap_or_else(|| "Hello, world".to_string());

    // Diagnostic: local GGUF parse + local encode (also the fallback when no
    // server is reachable).
    match gguf::parse_path(&gguf_path) {
        Ok(vocab) => {
            println!("vocab: model={:?} pre={:?} tokens={} merges={} types={}",
                vocab.model, vocab.pre, vocab.tokens.len(), vocab.merges.len(), vocab.token_types.len());
            if args.len() < 4 {
                let tok = match Tokenizer::from_vocab(&vocab) {
                    Ok(t) => t,
                    Err(e) => { eprintln!("tokenizer error: {e}"); std::process::exit(1); }
                };
                let ids = tok.encode(text.as_bytes());
                println!("local-encoded \"{text}\" -> {} ids: {:?}", ids.len(), &ids[..ids.len().min(64)]);
            }
        }
        Err(e) => eprintln!("(no local gguf parse: {e})"),
    }

    if args.len() >= 4 {
        let host = &args[3]; // e.g. 192.168.2.64
        let port: u16 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(6464);
        let _model = args.get(5).cloned().unwrap_or_else(|| "/nzk/models/Ternary-Bonsai-27B-MTP-TQ2_0.gguf".to_string());

        // 1. Get the tokenizer from the server: POST /tokenize -> exact ids.
        println!("GET tokenizer from server {host}:{port} (/tokenize) ...");
        let ids = match pretok::tokenize_via_server(&host, port, &text, false, 120_000) {
            Ok(ids) => ids,
            Err(e) => { eprintln!("server tokenize failed: {e}"); std::process::exit(1); }
        };
        println!("server tokenized \"{text}\" -> {} ids: {:?}", ids.len(), &ids[..ids.len().min(64)]);

        // 2. Send the pretokenized array via the direct-token endpoint.
        let msg = pretok::Msg {
            role: "user".to_string(),
            parts: vec![pretok::Part::Text(text.clone()), pretok::Part::InputTokens(ids)],
        };
        println!("POST /v1/chat_pretokenized -> {host}:{port}");
        match pretok::chat_pretokenized(&host, port, "/nzk/models/Ternary-Bonsai-27B-MTP-TQ2_0.gguf", &[msg], 32, 0.0, false, 120_000) {
            Ok(r) => {
                let truncated: String = r.body.chars().take(300).collect();
                println!("status {}: {truncated}", r.status);
            }
            Err(e) => println!("request error: {e}"),
        }
    }
}
