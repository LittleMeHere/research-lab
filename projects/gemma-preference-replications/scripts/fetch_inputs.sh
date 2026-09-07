#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p external

fetch() {
    local dest="$1" url="$2" revision="$3"
    if [[ ! -e "$dest" ]]; then
        git clone --no-checkout "$url" "$dest"
        git -C "$dest" checkout --detach "$revision"
    fi
    [[ "$(git -C "$dest" rev-parse HEAD)" == "$revision" ]] || {
        echo "Wrong upstream revision in $dest; use a fresh rerun directory." >&2
        exit 1
    }
    [[ -z "$(git -C "$dest" status --porcelain)" ]] || {
        echo "Modified upstream inputs in $dest; use a clean checkout." >&2
        exit 1
    }
}

fetch external/gilg https://github.com/oscar-gilg/probing-persona-preferences 11869a5ef93a30f8d8856246f57ceeefdc9b3b1f
fetch external/value_leakage https://github.com/TruthfulAI-research/value_leakage f7e5480cfe8abeb64b7007ba24fb0164519c3b68
