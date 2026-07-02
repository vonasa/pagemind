#!/usr/bin/env bash
# Native (non-Docker) Infinity embedding server for PageMind.
#
# Serves infgrad/Jasper-Token-Compression-600M on the host (arm64, no QEMU, no
# 2 GiB Colima VM to OOM in). The app talks to it over HTTP at
# http://localhost:7997/v1/embeddings — see pagemind/models/embeddings.py.
#
# Idempotent: creates a dedicated venv + downloads the model on first run, then
# just launches. Stays in the FOREGROUND — run it in its own terminal; Ctrl-C to
# stop. Equivalent recipe: `just embed`.
#
# Flags that matter (each one fixes a real failure mode — see CLAUDE.local.md):
#   --engine torch          infinity's auto-select picks the optimum/onnx engine,
#                           which isn't installed and import-errors.
#   --no-bettertransformer  infinity references BetterTransformerManager (an
#                           optimum symbol) unless this is off; NameError otherwise.
#   --url-prefix /v1        infinity serves routes at the root by default; the app
#                           posts to /v1/embeddings.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.infinity/venv"
REQ="$ROOT/scripts/infinity-requirements.txt"
export HF_HOME="$ROOT/.infinity/hf_cache"
PORT="${INFINITY_PORT:-7997}"

if [[ ! -x "$VENV/bin/infinity_emb" ]]; then
  echo "==> Creating embedding venv at $VENV"
  uv venv "$VENV" --python 3.13
  echo "==> Installing pinned deps from $REQ (first run pulls torch; ~1 min)"
  uv pip install --python "$VENV" -r "$REQ"
fi

echo "==> Infinity starting on http://localhost:$PORT  (model cache: $HF_HOME)"
echo "    First run downloads the model (~2.2 GB); subsequent starts are fast."
exec "$VENV/bin/infinity_emb" v2 \
  --model-id infgrad/Jasper-Token-Compression-600M \
  --served-model-name Jasper-Token-Compression-600M \
  --engine torch --device cpu --no-bettertransformer \
  --url-prefix /v1 --port "$PORT"
