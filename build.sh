#!/usr/bin/env bash
# Build the gigatoken C-ABI tokenizer backend and this addon.
#
#   1. Copies vendor/gigatoken from the llama-direct-token-input fork.
#   2. Applies patches/gigatoken-llama-cpp.patch (adds the gt_llama_* C ABI).
#   3. Builds gigatoken as a cdylib with `--features llama-cpp`.
#   4. Builds this crate; at runtime the addon prefers gigatoken and falls
#      back to the pure-Rust byte-level BPE.
#
# The toolchain must be nightly; gigatoken pins nightly-2026-07-22 but works
# on the active default nightly here.
#
# Usage: ./build.sh
set -euo pipefail

FORK=/nzk/git/pithagoras/gitrepos/llama-direct-token-input
HERE="$(cd "$(dirname "$0")" && pwd)"
PATCH="$FORK/patches/gigatoken-llama-cpp.patch"
GTSRC="$HERE/gigatoken-src"
GTTARGET="$HERE/gigatoken-target"

echo "== 1. stage gigatoken source (patched) =="
if [ ! -f "$GTSRC/include/gigatoken_llama.h" ] || [ -n "${REBUILD_GT:-}" ]; then
    rm -rf "$GTSRC"
    mkdir -p "$GTSRC"
    cp -a "$FORK/vendor/gigatoken/." "$GTSRC/"
    rm -rf "$GTSRC/.git" "$GTSRC/target"
    (cd "$FORK" && git apply --unsafe-paths --directory="$GTSRC" "$PATCH")
    echo "   patched -> $GTSRC/include/gigatoken_llama.h"
fi

echo "== 2. build gigatoken cdylib (features=llama-cpp) =="
(cd "$GTSRC" && CARGO_TARGET_DIR="$GTTARGET" \
    cargo rustc --lib --release --locked --no-default-features \
    --features llama-cpp --crate-type cdylib)

echo "== 3. build this addon crate =="
(cd "$HERE" && cargo build --release)

echo "== built =="
echo "   gigatoken: $GTTARGET/release/libgigatoken_rs.so"
echo "   addon    : $HERE/target/release/libgigatoken_addon.so"
echo ""
echo "The addon loads gigatoken from GT_CANDIDATE_LIBS at runtime; export"
echo "BONSAI_GGUF to point it at the model GGUF (defaults to the 27B MTP file)."
