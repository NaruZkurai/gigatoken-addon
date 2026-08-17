//! CLI smoke test: parse the Bonsai GGUF, tokenize a string locally, and
//! optionally send it through the direct-token endpoint.
use gigatoken_addon::{gguf, pretok, Tokenizer};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let gguf_path = args.get(1).cloned().unwrap_or_else(|| {
        "/run/media/naruzkurai/Win-ntfs/Ternary-Bonsai-27B-MTP-TQ2_0.gguf".to_string()
    });
    let text = args.get(2).cloned().unwrap_or_else(|| "Hello, world".to_string());

    println!("parsing gguf: {gguf_path}");
    let vocab = match gguf::parse_path(&gguf_path) {
        Ok(v) => v,
        Err(e) => { eprintln!("parse error: {e}"); std::process::exit(1); }
    };
    println!("vocab: model={:?} pre={:?} tokens={} merges={} types={}",
        vocab.model, vocab.pre, vocab.tokens.len(), vocab.merges.len(), vocab.token_types.len());

    let tok = match Tokenizer::from_vocab(&vocab) {
        Ok(t) => t,
        Err(e) => { eprintln!("tokenizer error: {e}"); std::process::exit(1); }
    };

    let ids = tok.encode(text.as_bytes());
    println!("encoded \"{text}\" -> {} ids: {:?}", ids.len(), &ids[..ids.len().min(64)]);

    if args.len() >= 4 {
        let host = &args[3]; // e.g. 192.168.2.64
        let port: u16 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(6464);
        let model = args.get(5).cloned().unwrap_or_else(|| "/nzk/models/Ternary-Bonsai-27B-MTP-TQ2_0.gguf".to_string());
        let msg = pretok::Msg {
            role: "user".to_string(),
            parts: vec![pretok::Part::Text(text.clone()), pretok::Part::InputTokens(ids)],
        };
        println!("POST /v1/chat_pretokenized -> {host}:{port} model={model}");
        match pretok::chat_pretokenized(&host, port, &model, &[msg], 32, 0.0, false, 120_000) {
            Ok(r) => {
                let truncated: String = r.body.chars().take(300).collect();
                println!("status {}: {truncated}", r.status);
            }
            Err(e) => println!("request error: {e}"),
        }
    }
}
