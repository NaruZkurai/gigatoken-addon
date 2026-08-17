//! CLI: the local-tokenization flow against a model server.
//!
//! 1. Pull the server's tokenizer.json (GET /v1/chat_tokenizer).
//! 2. Load it locally with gigatoken (a "personal tokenizer").
//! 3. Tokenize the prompt locally -> exact ids.
//! 4. Send the pre-tokenized array via POST /v1/chat_pretokenized.
//!
//! Because tokenization happens locally, the client sends raw ids and the
//! model server never runs its tokenizer over the prompt (saves compute).
//!
//! Usage: gt_cli <host> <port> [text] [model]
use gigatoken_addon::{hftok::LocalHfTokenizer, pretok};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let host = args.get(1).cloned().unwrap_or_else(|| "192.168.2.64".to_string());
    let port: u16 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(6464);
    let text = args.get(3).cloned().unwrap_or_else(|| "Hello, world!".to_string());
    let model = args.get(4).cloned().unwrap_or_else(|| "/nzk/models/Ternary-Bonsai-27B-MTP-TQ2_0.gguf".to_string());

    // 1. GET the tokenizer.json from the server.
    println!("1. fetching tokenizer.json from {host}:{port} (/v1/chat_tokenizer) ...");
    let mut tok = match LocalHfTokenizer::from_server(&host, port, 120_000) {
        Ok(t) => t,
        Err(e) => { eprintln!("tokenizer fetch/load failed: {e}"); std::process::exit(1); }
    };
    println!("   tokenizer loaded locally (gigatoken)");

    // 2+3. Tokenize locally with the personal tokenizer.
    let ids = tok.encode(&text);
    println!("2. local tokenize \"{text}\" -> {} ids: {:?}", ids.len(), &ids[..ids.len().min(64)]);

    // 4. Send the pre-tokenized array via the direct-token endpoint.
    let msg = pretok::Msg {
        role: "user".to_string(),
        parts: vec![pretok::Part::InputTokens(ids)],
    };
    println!("3. POST /v1/chat_pretokenized ({host}:{port}) ...");
    match pretok::chat_pretokenized(&host, port, &model, &[msg], 32, 0.0, false, 120_000) {
        Ok(r) => {
            let truncated: String = r.body.chars().take(300).collect();
            println!("   status {}: {truncated}", r.status);
        }
        Err(e) => println!("   request error: {e}"),
    }
}
