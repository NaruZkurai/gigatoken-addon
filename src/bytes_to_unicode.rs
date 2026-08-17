//! GPT-2 `bytes_to_unicode` reverse table.
//!
//! llama.cpp (and gigatoken) represent byte-level-BPE byte tokens by their raw
//! single byte, but Bonsai/qwen35 GGUF files store the 256 byte-fallback tokens
//! transcoded to unicode chars via the standard GPT-2 bytes_to_unicode mapping.
//! When handing the vocabulary to gigatoken, each byte token's `text` must be
//! the single raw byte, so we map the stored unicode UTF-8 back to the byte.
//!
//! @module gigatoken-addon/bytes_to_unicode

/// Map one of the 256 GPT-2 byte-fallback unicode chars (as UTF-8 bytes) back
/// to its raw byte. Returns `None` when `text` is not a byte-token char.
pub fn transcode_byte_token(text: &[u8]) -> Option<u8> {
    let b = match text {
        b"\xc4\x80" => 0, b"\xc4\x81" => 1, b"\xc4\x82" => 2, b"\xc4\x83" => 3, b"\xc4\x84" => 4,
        b"\xc4\x85" => 5, b"\xc4\x86" => 6, b"\xc4\x87" => 7, b"\xc4\x88" => 8, b"\xc4\x89" => 9,
        b"\xc4\x8a" => 10, b"\xc4\x8b" => 11, b"\xc4\x8c" => 12, b"\xc4\x8d" => 13, b"\xc4\x8e" => 14,
        b"\xc4\x8f" => 15, b"\xc4\x90" => 16, b"\xc4\x91" => 17, b"\xc4\x92" => 18, b"\xc4\x93" => 19,
        b"\xc4\x94" => 20, b"\xc4\x95" => 21, b"\xc4\x96" => 22, b"\xc4\x97" => 23, b"\xc4\x98" => 24,
        b"\xc4\x99" => 25, b"\xc4\x9a" => 26, b"\xc4\x9b" => 27, b"\xc4\x9c" => 28, b"\xc4\x9d" => 29,
        b"\xc4\x9e" => 30, b"\xc4\x9f" => 31, b"\xc4\xa0" => 32,
        b"!" => 33, b"\"" => 34, b"#" => 35, b"$" => 36, b"%" => 37, b"&" => 38, b"'" => 39,
        b"(" => 40, b")" => 41, b"*" => 42, b"+" => 43, b"," => 44, b"-" => 45, b"." => 46,
        b"/" => 47, b"0" => 48, b"1" => 49, b"2" => 50, b"3" => 51, b"4" => 52, b"5" => 53,
        b"6" => 54, b"7" => 55, b"8" => 56, b"9" => 57, b":" => 58, b";" => 59, b"<" => 60,
        b"=" => 61, b">" => 62, b"?" => 63, b"@" => 64,
        b"A" => 65, b"B" => 66, b"C" => 67, b"D" => 68, b"E" => 69, b"F" => 70, b"G" => 71,
        b"H" => 72, b"I" => 73, b"J" => 74, b"K" => 75, b"L" => 76, b"M" => 77, b"N" => 78,
        b"O" => 79, b"P" => 80, b"Q" => 81, b"R" => 82, b"S" => 83, b"T" => 84, b"U" => 85,
        b"V" => 86, b"W" => 87, b"X" => 88, b"Y" => 89, b"Z" => 90, b"[" => 91, b"\\" => 92,
        b"]" => 93, b"^" => 94, b"_" => 95, b"`" => 96,
        b"a" => 97, b"b" => 98, b"c" => 99, b"d" => 100, b"e" => 101, b"f" => 102, b"g" => 103,
        b"h" => 104, b"i" => 105, b"j" => 106, b"k" => 107, b"l" => 108, b"m" => 109, b"n" => 110,
        b"o" => 111, b"p" => 112, b"q" => 113, b"r" => 114, b"s" => 115, b"t" => 116, b"u" => 117,
        b"v" => 118, b"w" => 119, b"x" => 120, b"y" => 121, b"z" => 122, b"{" => 123, b"|" => 124,
        b"}" => 125, b"~" => 126,
        b"\xc4\xa1" => 127, b"\xc4\xa2" => 128, b"\xc4\xa3" => 129, b"\xc4\xa4" => 130, b"\xc4\xa5" => 131,
        b"\xc4\xa6" => 132, b"\xc4\xa7" => 133, b"\xc4\xa8" => 134, b"\xc4\xa9" => 135, b"\xc4\xaa" => 136,
        b"\xc4\xab" => 137, b"\xc4\xac" => 138, b"\xc4\xad" => 139, b"\xc4\xae" => 140, b"\xc4\xaf" => 141,
        b"\xc4\xb0" => 142, b"\xc4\xb1" => 143, b"\xc4\xb2" => 144, b"\xc4\xb3" => 145, b"\xc4\xb4" => 146,
        b"\xc4\xb5" => 147, b"\xc4\xb6" => 148, b"\xc4\xb7" => 149, b"\xc4\xb8" => 150, b"\xc4\xb9" => 151,
        b"\xc4\xba" => 152, b"\xc4\xbb" => 153, b"\xc4\xbc" => 154, b"\xc4\xbd" => 155, b"\xc4\xbe" => 156,
        b"\xc4\xbf" => 157, b"\xc5\x80" => 158, b"\xc5\x81" => 159, b"\xc5\x82" => 160,
        b"\xc2\xa1" => 161, b"\xc2\xa2" => 162, b"\xc2\xa3" => 163, b"\xc2\xa4" => 164, b"\xc2\xa5" => 165,
        b"\xc2\xa6" => 166, b"\xc2\xa7" => 167, b"\xc2\xa8" => 168, b"\xc2\xa9" => 169, b"\xc2\xaa" => 170,
        b"\xc2\xab" => 171, b"\xc2\xac" => 172, b"\xc5\x83" => 173, b"\xc2\xae" => 174, b"\xc2\xaf" => 175,
        b"\xc2\xb0" => 176, b"\xc2\xb1" => 177, b"\xc2\xb2" => 178, b"\xc2\xb3" => 179, b"\xc2\xb4" => 180,
        b"\xc2\xb5" => 181, b"\xc2\xb6" => 182, b"\xc2\xb7" => 183, b"\xc2\xb8" => 184, b"\xc2\xb9" => 185,
        b"\xc2\xba" => 186, b"\xc2\xbb" => 187, b"\xc2\xbc" => 188, b"\xc2\xbd" => 189, b"\xc2\xbe" => 190,
        b"\xc2\xbf" => 191,
        b"\xc3\x80" => 192, b"\xc3\x81" => 193, b"\xc3\x82" => 194, b"\xc3\x83" => 195, b"\xc3\x84" => 196,
        b"\xc3\x85" => 197, b"\xc3\x86" => 198, b"\xc3\x87" => 199, b"\xc3\x88" => 200, b"\xc3\x89" => 201,
        b"\xc3\x8a" => 202, b"\xc3\x8b" => 203, b"\xc3\x8c" => 204, b"\xc3\x8d" => 205, b"\xc3\x8e" => 206,
        b"\xc3\x8f" => 207, b"\xc3\x90" => 208, b"\xc3\x91" => 209, b"\xc3\x92" => 210, b"\xc3\x93" => 211,
        b"\xc3\x94" => 212, b"\xc3\x95" => 213, b"\xc3\x96" => 214, b"\xc3\x97" => 215, b"\xc3\x98" => 216,
        b"\xc3\x99" => 217, b"\xc3\x9a" => 218, b"\xc3\x9b" => 219, b"\xc3\x9c" => 220, b"\xc3\x9d" => 221,
        b"\xc3\x9e" => 222, b"\xc3\x9f" => 223, b"\xc3\xa0" => 224, b"\xc3\xa1" => 225, b"\xc3\xa2" => 226,
        b"\xc3\xa3" => 227, b"\xc3\xa4" => 228, b"\xc3\xa5" => 229, b"\xc3\xa6" => 230, b"\xc3\xa7" => 231,
        b"\xc3\xa8" => 232, b"\xc3\xa9" => 233, b"\xc3\xaa" => 234, b"\xc3\xab" => 235, b"\xc3\xac" => 236,
        b"\xc3\xad" => 237, b"\xc3\xae" => 238, b"\xc3\xaf" => 239, b"\xc3\xb0" => 240, b"\xc3\xb1" => 241,
        b"\xc3\xb2" => 242, b"\xc3\xb3" => 243, b"\xc3\xb4" => 244, b"\xc3\xb5" => 245, b"\xc3\xb6" => 246,
        b"\xc3\xb7" => 247, b"\xc3\xb8" => 248, b"\xc3\xb9" => 249, b"\xc3\xba" => 250, b"\xc3\xbb" => 251,
        b"\xc3\xbc" => 252, b"\xc3\xbd" => 253, b"\xc3\xbe" => 254, b"\xc3\xbf" => 255,
        _ => return None,
    };
    Some(b)
}
