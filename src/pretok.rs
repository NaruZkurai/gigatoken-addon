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

/// Parse the HTTP response into status + body string.
///
/// Handles the three legitimate body framings properly so JSON bodies are
/// returned intact:
///   * `Content-Length` — read exactly that many body bytes.
///   * `Transfer-Encoding: chunked` — de-chunk the body.
///   * otherwise — read to EOF (relying on `Connection: close`).
fn read_response(stream: &mut TcpStream) -> Result<(u16, String), String> {
    // 1. Read the header block (ending at the blank line).
    let mut raw: Vec<u8> = Vec::new();
    let mut buf = [0u8; 8192];
    let header_end = loop {
        if let Some(i) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        let n = stream.read(&mut buf).map_err(|e| format!("read headers: {e}"))?;
        if n == 0 {
            return Err("connection closed before headers completed".into());
        }
        raw.extend_from_slice(&buf[..n]);
    };

    let head = String::from_utf8_lossy(&raw[..header_end]);
    let mut lines = head.lines();
    let status: u16 = lines
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    // Gather headers (case-insensitive keys).
    let mut content_length: Option<usize> = None;
    let mut chunked = false;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            let key = k.trim().to_ascii_lowercase();
            let val = v.trim().to_ascii_lowercase();
            match key.as_str() {
                "content-length" => content_length = val.parse().ok(),
                "transfer-encoding" => {
                    if val.contains("chunked") {
                        chunked = true;
                    }
                }
                _ => {}
            }
        }
    }

    // 2. Read the body per its framing.
    let mut body: Vec<u8> = raw[header_end..].to_vec();
    if chunked {
        body = dechunk(&body, stream)?;
    } else if let Some(len) = content_length {
        while body.len() < len {
            let want = (len - body.len()).min(buf.len());
            let n = stream.read(&mut buf[..want]).map_err(|e| format!("read body: {e}"))?;
            if n == 0 {
                break; // short body; return what we got
            }
            body.extend_from_slice(&buf[..n]);
        }
        body.truncate(len);
    } else {
        // No framing: read to EOF (server uses Connection: close).
        loop {
            let n = stream.read(&mut buf).map_err(|e| format!("read body: {e}"))?;
            if n == 0 {
                break;
            }
            body.extend_from_slice(&buf[..n]);
        }
    }

    let text = String::from_utf8_lossy(&body).into_owned();
    Ok((status, text))
}

/// De-chunk an HTTP `Transfer-Encoding: chunked` body. `prefix` may carry body
/// bytes that arrived with the headers; remaining chunks are read from the
/// stream. Returns the concatenated plain body.
fn dechunk(prefix: &[u8], stream: &mut TcpStream) -> Result<Vec<u8>, String> {
    let mut raw: Vec<u8> = prefix.to_vec(); // unconsumed stream buffer
    let mut pos = 0usize;                   // read cursor into `raw`
    let mut out: Vec<u8> = Vec::new();
    let mut buf = [0u8; 8192];

    /// Read more bytes into `raw` until at least `need` unconsumed bytes exist.
    fn fill(raw: &mut Vec<u8>, pos: usize, need: usize, stream: &mut TcpStream, buf: &mut [u8; 8192]) -> Result<(), String> {
        while raw.len() - pos < need {
            let n = stream.read(buf).map_err(|e| format!("read chunk: {e}"))?;
            if n == 0 {
                return Err(format!("connection closed mid-chunk (need {need}, have {})", raw.len() - pos));
            }
            raw.extend_from_slice(&buf[..n]);
        }
        Ok(())
    }

    loop {
        // Read the chunk-size line (ends with CRLF).
        let mut crlf = raw[pos..].windows(2).position(|w| w == b"\r\n").map(|i| i + pos);
        while crlf.is_none() {
            let n = stream.read(&mut buf).map_err(|e| format!("read chunk size: {e}"))?;
            if n == 0 {
                return Err("connection closed before chunk-size line".into());
            }
            raw.extend_from_slice(&buf[..n]);
            crlf = raw[pos..].windows(2).position(|w| w == b"\r\n").map(|i| i + pos);
        }
        let crlf = crlf.unwrap();
        let line = &raw[pos..crlf];
        let size_field = line.split(|&b| b == b';').next().unwrap_or(line);
        let size = usize::from_str_radix(String::from_utf8_lossy(size_field).trim(), 16)
            .map_err(|e| format!("bad chunk size: {e}"))?;
        pos = crlf + 2; // move past the size line + CRLF
        if size == 0 {
            break; // terminal chunk
        }
        // Ensure the chunk data + trailing CRLF are available.
        fill(&mut raw, pos, size + 2, stream, &mut buf)?;
        // Copy the chunk data out and compact the buffer.
        out.extend_from_slice(&raw[pos..pos + size]);
        pos += size + 2; // skip chunk data + its CRLF
        if pos > 1 << 20 {
            raw.drain(..pos);
            pos = 0;
        }
    }
    Ok(out)
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

    // Use the shared framing-aware reader so chunked/Content-Length bodies are
    // returned intact (the tokenizer.json is several MB and can be chunked).
    let (status, body) = read_response(&mut stream)?;
    if status != 200 {
        return Err(format!("chat_tokenizer returned {status}: {body}"));
    }
    Ok(body.into_bytes())
}

fn parse_input_tokens(body: &str) -> Option<u64> {
    let key = "\"input_tokens\"";
    let idx = body.find(key)?;
    let rest = &body[idx + key.len()..];
    let start = rest.find(':')? + 1;
    let end = rest[start..].find([',', '}']).map(|e| start + e).unwrap_or(rest.len());
    rest[start..end].trim().parse().ok()
}
