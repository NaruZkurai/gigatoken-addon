//! Minimal OpenAI-compatible direct-token client for the fork's
//! `/v1/chat_pretokenized` endpoint, implemented over `std::net::TcpStream`
//! with no external dependencies.
//!
//! The request body is a normal chat-completions body in which message content
//! may be replaced with the direct-token parts form:
//!   content: [ { "type":"text", "text":"..." },
//!              { "type":"input_tokens", "tokens":[123, 456, ...] } ]
//! (or a whole-content array of raw ints). The server renders the chat template
//! around them and splices the raw ids into the prompt, so the heavy text has
//! already been tokenized locally by this addon.
//!
//! @module gigatoken-addon/pretok

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

/// A message part for the direct-token body.
pub enum Part {
    /// Plain text to be tokenized by the server (use sparingly).
    Text(String),
    /// A locally-tokenized run of ids sent via the direct-token API.
    InputTokens(Vec<u32>),
}

/// A chat message to send through the direct-token endpoint.
pub struct Msg {
    pub role: String,
    pub parts: Vec<Part>,
}

fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

fn render_parts(parts: &[Part], out: &mut String) {
    // If a message is a single InputTokens run, emit the whole content as an
    // int array (server treats array-of-int content as raw tokens).
    if parts.len() == 1 {
        if let Part::InputTokens(tokens) = &parts[0] {
            out.push_str("[");
            for (i, t) in tokens.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&t.to_string());
            }
            out.push_str("]");
            return;
        }
    }
    // Otherwise render the parts-array JSON form.
    out.push_str("[");
    for (i, p) in parts.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        match p {
            Part::Text(t) => {
                out.push_str(&format!("{{\"type\":\"text\",\"text\":\"{}\"}}", escape_json_string(t)));
            }
            Part::InputTokens(tokens) => {
                out.push_str("{\"type\":\"input_tokens\",\"tokens\":[");
                for (j, t) in tokens.iter().enumerate() {
                    if j > 0 {
                        out.push(',');
                    }
                    out.push_str(&t.to_string());
                }
                out.push_str("]}");
            }
        }
    }
    out.push_str("]");
}

/// Render the full JSON body for a direct-token request.
pub fn render_body(
    model: &str,
    messages: &[Msg],
    max_tokens: u32,
    temperature: f32,
    stream: bool,
) -> String {
    let mut body = String::new();
    body.push_str(&format!("{{\"model\":\"{}\",\"messages\":[", escape_json_string(model)));
    for (i, m) in messages.iter().enumerate() {
        if i > 0 {
            body.push(',');
        }
        body.push_str(&format!("{{\"role\":\"{}\",\"content\":", escape_json_string(&m.role)));
        render_parts(&m.parts, &mut body);
        body.push('}');
    }
    body.push_str(&format!(
        "],\"max_tokens\":{},\"temperature\":{},\"stream\":{}",
        max_tokens,
        temperature,
        if stream { "true" } else { "false" }
    ));
    body.push('}');
    body
}

/// Parse the HTTP response into status + optional body.
fn read_response(stream: &mut TcpStream) -> Result<(u16, String), String> {
    let mut raw = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = stream.read(&mut buf).map_err(|e| format!("read: {e}"))?;
        if n == 0 {
            break;
        }
        raw.extend_from_slice(&buf[..n]);
        if raw.windows(4).any(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&raw);
            let headers_end = head.find("\r\n\r\n").unwrap() + 4;
            let lower = head.to_ascii_lowercase();
            // We read until we have the full body for a non-streaming reply; for
            // streaming we stop at the headers (caller can re-read chunks).
            let _ = headers_end;
            let has_encoding = lower.contains("transfer-encoding: chunked") || lower.contains("content-length");
            if !has_encoding {
                break;
            }
            // For a bounded body we can attempt to read the rest; for SSE we
            // break on headers and leave the body to the caller.
            if lower.contains("text/event-stream") {
                break;
            }
        }
    }

    let head = String::from_utf8_lossy(&raw);
    let line = head.lines().next().unwrap_or("");
    let status: u16 = line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    // Split off body (everything after the blank line).
    let body = match head.find("\r\n\r\n") {
        Some(i) => head[i + 4..].to_string(),
        None => head.clone().into_owned(),
    };
    Ok((status, body))
}

/// A completed (non-stream) response.
pub struct Completion {
    pub status: u16,
    pub body: String,
}

/// POST a direct-token request to `/v1/chat_pretokenized`.
///
/// `host`/`port` are the server address; `path` defaults to
/// `/v1/chat_pretokenized`. A dummy `Authorization` header is sent (the server
/// ignores it but pi-ai/harness-style clients set one).
pub fn chat_pretokenized(
    host: &str,
    port: u16,
    model: &str,
    messages: &[Msg],
    max_tokens: u32,
    temperature: f32,
    stream: bool,
    timeout_ms: u64,
) -> Result<Completion, String> {
    let body = render_body(model, messages, max_tokens, temperature, stream);
    let path = "/v1/chat_pretokenized";

    let mut stream = TcpStream::connect((host, port)).map_err(|e| format!("connect {host}:{port}: {e}"))?;
    stream.set_read_timeout(Some(Duration::from_millis(timeout_ms))).ok();
    stream.set_write_timeout(Some(Duration::from_millis(timeout_ms))).ok();

    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\nAuthorization: Bearer local-keyless\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|e| format!("write: {e}"))?;

    let (status, response_body) = read_response(&mut stream)?;
    Ok(Completion { status, body: response_body })
}

/// Count tokens for a fully-tokenized list via `/v1/chat/completions/input_tokens`.
pub fn count_input_tokens(host: &str, port: u16, model: &str, tokens: &[u32], timeout_ms: u64) -> Result<u64, String> {
    let mut body = String::new();
    body.push_str(&format!("{{\"model\":\"{}\",\"input_tokens\":[", escape_json_string(model)));
    for (i, t) in tokens.iter().enumerate() {
        if i > 0 {
            body.push(',');
        }
        body.push_str(&t.to_string());
    }
    body.push_str("]}");
    let path = "/v1/chat/completions/input_tokens";

    let mut stream = TcpStream::connect((host, port)).map_err(|e| format!("connect: {e}"))?;
    stream.set_read_timeout(Some(Duration::from_millis(timeout_ms))).ok();
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\nAuthorization: Bearer local-keyless\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(request.as_bytes()).map_err(|e| format!("write: {e}"))?;
    let (status, response_body) = read_response(&mut stream)?;
    if status != 200 {
        return Err(format!("count returned {status}: {response_body}"));
    }
    // Response: {"input_tokens": N, ...}
    Ok(parse_input_tokens(&response_body).unwrap_or(0))
}

/// Tokenize text through the model server's own tokenizer (`POST /tokenize`).
///
/// This is the authoritative tokenizer for a server-served model: the ids it
/// returns are byte-exact by construction, so the direct-token prompts built
/// from them are guaranteed correct. Use this instead of a local reconstruction
/// whenever the model is loaded on a reachable fork server.
///
/// `content` may also be a raw token array to round-trip (the server accepts a
/// mixed text/ids array), but callers want plain text here. `add_special` and
/// `parse_special` mirror the server flags; `with_pieces` is left off (we only
/// need ids).
pub fn tokenize_via_server(
    host: &str,
    port: u16,
    content: &str,
    add_special: bool,
    timeout_ms: u64,
) -> Result<Vec<u32>, String> {
    let body = format!(
        "{{\"content\":\"{}\",\"add_special\":{},\"parse_special\":true}}",
        escape_json_string(content),
        if add_special { "true" } else { "false" }
    );
    let path = "/tokenize";

    let mut stream = TcpStream::connect((host, port)).map_err(|e| format!("connect {host}:{port}: {e}"))?;
    stream.set_read_timeout(Some(Duration::from_millis(timeout_ms))).ok();
    stream.set_write_timeout(Some(Duration::from_millis(timeout_ms))).ok();

    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\nAuthorization: Bearer local-keyless\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(request.as_bytes()).map_err(|e| format!("write: {e}"))?;
    let (status, response_body) = read_response(&mut stream)?;
    if status != 200 {
        return Err(format!("tokenize returned {status}: {response_body}"));
    }
    parse_tokens_json(&response_body).ok_or_else(|| format!("could not parse tokens from: {response_body}"))
}

/// Parse the `{"tokens":[...]}` JSON response into a `Vec<u32>`.
fn parse_tokens_json(body: &str) -> Option<Vec<u32>> {
    let key = "\"tokens\"";
    let idx = body.find(key)?;
    let rest = &body[idx + key.len()..];
    let start = rest.find('[')? + 1;
    let end = rest[start..].find(']').map(|e| start + e)?;
    let mut out = Vec::new();
    for part in rest[start..end].split(',') {
        let t = part.trim().parse::<u32>().ok()?;
        out.push(t);
    }
    Some(out)
}

/// Fetch the model server's HugoayFace-compatible `tokenizer.json`
/// (`GET /v1/chat_tokenizer`), returning the raw JSON bytes.
///
/// This is the byte-exact source of truth for the loaded model and is what a
/// local tokenizer (gigatoken `load_hf_slice`) consumes to reproduce the
/// server's exact token ids — enabling fully local tokenization so the server
/// never processes the prompt text.
pub fn fetch_tokenizer_json(host: &str, port: u16, timeout_ms: u64) -> Result<Vec<u8>, String> {
    let path = "/v1/chat_tokenizer";
    let mut stream = TcpStream::connect((host, port)).map_err(|e| format!("connect {host}:{port}: {e}"))?;
    stream.set_read_timeout(Some(Duration::from_millis(timeout_ms))).ok();
    stream.set_write_timeout(Some(Duration::from_millis(timeout_ms))).ok();

    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nAuthorization: Bearer local-keyless\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(request.as_bytes()).map_err(|e| format!("write: {e}"))?;

    // Read the whole response; the body is the tokenizer JSON (a few MB).
    let mut raw = Vec::new();
    let mut buf = [0u8; 65536];
    loop {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => raw.extend_from_slice(&buf[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock || e.kind() == std::io::ErrorKind::TimedOut => break,
            Err(e) => return Err(format!("read: {e}")),
        }
    }
    let text = String::from_utf8_lossy(&raw);
    let head_end = text.find("\r\n\r\n").unwrap_or(0);
    // Caller only gets the body; strip the HTTP head.
    let body_start = if head_end == 0 { 0 } else { head_end + 4 };
    let body = &raw[raw.len().min(body_start)..];
    Ok(body.to_vec())
}

fn parse_input_tokens(body: &str) -> Option<u64> {
    let key = "\"input_tokens\"";
    let idx = body.find(key)?;
    let rest = &body[idx + key.len()..];
    let start = rest.find(':')? + 1;
    let end = rest[start..].find([',', '}']).map(|e| start + e).unwrap_or(rest.len());
    rest[start..end].trim().parse().ok()
}
