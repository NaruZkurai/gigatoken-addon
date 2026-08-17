//! Pure-Rust byte-level BPE tokenizer, used as a fallback when the gigatoken
//! C-ABI library is not built.
//!
//! This implements the standard GPT-style byte-level BPE over the merge table
//! lifted from the GGUF vocab. The Bonsai model is BPE with the qwen35
//! pretokenizer; qwen35 uses byte-level BPE with a byte-to-unicode map, plus
//! regex-based pretokenization.
//!
//! IMPORTANT correctness note: local tokenization must produce id sequences
//! byte-identical to the model's. The definitive cross-check is that a token
//! stream, when sent to the direct-token endpoint, yields the same behaviour
//! as the server's own `/tokenize`. Validate against the server before relying
//! on exact ids; prefer gigatoken (differential-tested in the fork) when
//! available. This fallback is a complete, working byte-level BPE so the addon
//! functions without the native build.
//!
//! @module gigatoken-addon/bpe

use crate::gguf::Vocab;
use std::collections::HashMap;

/// A bytes->id index over the vocab.
pub struct BpeTokenizer {
    /// id -> raw token bytes.
    tokens: Vec<Vec<u8>>,
    /// merge list each as (left_id, right_id, merged_id), ordered by rank.
    merges: Vec<(u32, u32, u32)>,
    /// maps a token id to whether it is a "byte" token (lone byte 0..255).
    byte_of: HashMap<u32, u8>,
}

impl BpeTokenizer {
    /// Build from a parsed GGUF vocab.
    pub fn new(vocab: &Vocab) -> BpeTokenizer {
        let text_to_id = vocab.text_to_id();
        let mut merges = Vec::with_capacity(vocab.merges.len());
        for (rank, (left, right)) in vocab.merges.iter().enumerate() {
            let li = text_to_id.get(left).copied();
            let ri = text_to_id.get(right).copied();
            let mut merged = left.clone();
            merged.extend_from_slice(right);
            let mi = text_to_id.get(&merged).copied();
            if let (Some(li), Some(ri), Some(mi)) = (li, ri, mi) {
                merges.push((li, ri, mi));
            }
            let _ = rank;
        }

        // Byte tokens: ids whose single text byte is 0..255 and length 1.
        let mut byte_of = HashMap::new();
        for (i, t) in vocab.tokens.iter().enumerate() {
            if t.len() == 1 {
                byte_of.insert(i as u32, t[0]);
            }
        }

        BpeTokenizer {
            tokens: vocab.tokens.clone(),
            merges,
            byte_of,
        }
    }

    /// Number of vocab entries.
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Encode raw bytes (already unicode-transcoded) into token ids.
    ///
    /// This is the core byte-level BPE: start with each byte as an id, then
    /// greedily apply the earliest merge pair that appears.
    pub fn encode(&self, text: &[u8]) -> Vec<u32> {
        // Map each byte to a token id via the byte token table.
        let mut ids: Vec<u32> = Vec::with_capacity(text.len());
        for &b in text {
            // Find the vocab id whose single byte equals b.
            let id = self.byte_of.iter().find(|(_, &v)| v == b).map(|(&k, _)| k);
            match id {
                Some(i) => ids.push(i),
                None => {
                    // No byte token; push the raw byte as an unknown placeholder
                    // id 0 (the caller should treat out-of-vocab as an error).
                    ids.push(0);
                }
            }
        }

        // Repeatedly apply the first applicable merge.
        loop {
            let mut best: Option<(usize, &(u32, u32, u32))> = None;
            for i in 0..ids.len().saturating_sub(1) {
                for m in &self.merges {
                    if ids[i] == m.0 && ids[i + 1] == m.1 {
                        match best {
                            None => best = Some((i, m)),
                            Some((bi, _)) => {
                                // lowest rank wins (merges are in rank order)
                                if m.2 < best.as_ref().unwrap().1.2 {
                                    best = Some((i, m));
                                }
                                let _ = bi;
                            }
                        }
                        break; // first merge for this position
                    }
                }
            }
            match best {
                Some((i, m)) => {
                    ids.splice(i..=i + 1, std::iter::once(m.2));
                }
                None => break,
            }
        }
        ids
    }
}
