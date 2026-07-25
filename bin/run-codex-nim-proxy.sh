#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v codex-nim-proxy >/dev/null 2>&1; then
    BINARY="codex-nim-proxy"
elif [[ -x "$SCRIPT_DIR/codex-nim-proxy" ]]; then
    BINARY="$SCRIPT_DIR/codex-nim-proxy"
else
    echo "ERROR: codex-nim-proxy binary not found." >&2
    exit 1
fi

if [[ -z "${NVIDIA_API_KEY:-}" ]] && [[ "$*" != *"--api-key"* ]]; then
    echo "WARNING: NVIDIA_API_KEY is not set." >&2
fi

exec "$BINARY" "$@"
