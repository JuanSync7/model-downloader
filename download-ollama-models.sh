#!/usr/bin/env bash
# Ollama Model Download & Update Script
# Downloads: qwen2.5:3b (LLM for RAG generation + query processing)
# Safe to re-run — only pulls if a newer version is available.
#
# Run with: bash ~/model-downloader/download-ollama-models.sh

set -e
LOGFILE="$HOME/download-ollama-models.log"
exec > >(tee -a "$LOGFILE") 2>&1

# Models to pull — add more entries here as needed
MODELS=(
  "qwen2.5:3b"
)

echo "============================================"
echo " Ollama Model Download / Update"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""

# ── Install / update ollama binary (user-local, no sudo) ─────────────────────
OLLAMA_BIN="$HOME/.local/bin/ollama"
OLLAMA_RELEASE_BASE="https://github.com/ollama/ollama/releases/download"
# Last release that ships .tgz (no zstd needed). Used as fallback.
OLLAMA_LAST_TGZ_VER="v0.13.0"

install_ollama() {
  local version="$1"
  local tmpdir
  tmpdir=$(mktemp -d)
  trap "rm -rf $tmpdir" RETURN

  mkdir -p "$(dirname "$OLLAMA_BIN")"

  local zst_url="${OLLAMA_RELEASE_BASE}/${version}/ollama-linux-amd64.tar.zst"
  local tgz_url="${OLLAMA_RELEASE_BASE}/${version}/ollama-linux-amd64.tgz"

  if command -v zstd &>/dev/null; then
    # zstd available — download latest .tar.zst
    echo "  Downloading ollama ${version} (.tar.zst)..."
    curl -fsSL --progress-bar "$zst_url" | zstd -d | tar -xf - -C "$tmpdir"
  elif curl -fsSL --head "$tgz_url" &>/dev/null 2>&1; then
    # No zstd but .tgz exists for this version
    echo "  Downloading ollama ${version} (.tgz)..."
    curl -fsSL --progress-bar "$tgz_url" | tar -xzf - -C "$tmpdir"
  else
    # No zstd and version too new for .tgz — fall back to last tgz release
    echo "  WARNING: zstd not found and ${version} only ships .tar.zst"
    echo "  Falling back to ${OLLAMA_LAST_TGZ_VER} (last .tgz release)"
    echo "  To get the latest version, install zstd: sudo apt-get install zstd"
    local fallback_url="${OLLAMA_RELEASE_BASE}/${OLLAMA_LAST_TGZ_VER}/ollama-linux-amd64.tgz"
    echo "  Downloading ollama ${OLLAMA_LAST_TGZ_VER} (.tgz)..."
    curl -fsSL --progress-bar "$fallback_url" | tar -xzf - -C "$tmpdir"
    version="$OLLAMA_LAST_TGZ_VER"
  fi

  # The archive extracts to bin/ollama
  if [ -f "$tmpdir/bin/ollama" ]; then
    mv "$tmpdir/bin/ollama" "$OLLAMA_BIN"
  elif [ -f "$tmpdir/ollama" ]; then
    mv "$tmpdir/ollama" "$OLLAMA_BIN"
  else
    echo "  ERROR: ollama binary not found in archive"
    return 1
  fi
  chmod +x "$OLLAMA_BIN"
  echo "  Installed version: $(\"$OLLAMA_BIN\" --version 2>&1 | awk '{print $NF}')"
}

# Fetch latest release tag
LATEST_TAG=$(curl -fsSL "https://api.github.com/repos/ollama/ollama/releases/latest" \
  | grep '"tag_name"' | head -1 | cut -d'"' -f4)
LATEST_VER="${LATEST_TAG#v}"
echo "  Latest Ollama release: $LATEST_VER"

if ! command -v ollama &>/dev/null; then
  echo "  Ollama not found — installing to $OLLAMA_BIN..."
  install_ollama "$LATEST_TAG"
  echo "  ✓ Ollama installed"
else
  OLLAMA_VERSION=$(ollama --version 2>&1 | awk '{print $NF}')
  echo "  Ollama $OLLAMA_VERSION installed at $(which ollama)"
  if [ "$OLLAMA_VERSION" = "$LATEST_VER" ]; then
    echo "  ✓ Ollama $OLLAMA_VERSION (up to date)"
  else
    echo "  ↑ Ollama $OLLAMA_VERSION → $LATEST_VER (upgrading)..."
    install_ollama "$LATEST_TAG"
    echo "  ✓ Ollama upgraded"
  fi
fi
echo ""

# ── Ensure ollama server is running ──────────────────────────────────────────
if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
  echo "  Starting ollama server..."
  OLLAMA_LOG=$(mktemp)
  ollama serve >"$OLLAMA_LOG" 2>&1 &
  OLLAMA_PID=$!
  STARTED_SERVER=true
  # Wait for server to be ready (up to 30s — first start can be slow)
  for i in {1..30}; do
    # Check if the process died
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
      echo "  ERROR: Ollama server exited unexpectedly. Logs:"
      cat "$OLLAMA_LOG" | sed 's/^/    /'
      rm -f "$OLLAMA_LOG"
      exit 1
    fi
    if curl -sf http://localhost:11434/api/tags &>/dev/null; then
      break
    fi
    sleep 1
  done
  rm -f "$OLLAMA_LOG"
  if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
    echo "  ERROR: Ollama server did not respond after 30s"
    echo "  Check if port 11434 is blocked or another process is using it"
    kill "$OLLAMA_PID" 2>/dev/null || true
    exit 1
  fi
  echo "  ✓ Ollama server started (PID $OLLAMA_PID)"
else
  STARTED_SERVER=false
  echo "  ✓ Ollama server already running"
fi
echo ""

# ── Pull models ──────────────────────────────────────────────────────────────
TOTAL=${#MODELS[@]}
STEP=0

for MODEL in "${MODELS[@]}"; do
  STEP=$((STEP + 1))
  echo "─────────────────────────────────────────────"
  echo " Step $STEP/$TOTAL — $MODEL"
  echo "─────────────────────────────────────────────"

  # Check if model already exists locally
  if ollama list 2>/dev/null | grep -q "^${MODEL}"; then
    echo "  Model exists locally, checking for updates..."
  else
    echo "  Model not found locally, downloading..."
  fi

  ollama pull "$MODEL"
  echo "  ✓ $MODEL ready"
  echo ""
done

# ── Verify models ────────────────────────────────────────────────────────────
echo "============================================"
echo " Verifying models..."
echo "============================================"
echo ""

ALL_OK=true
for MODEL in "${MODELS[@]}"; do
  if ollama list 2>/dev/null | grep -q "^${MODEL}"; then
    SIZE=$(ollama list 2>/dev/null | grep "^${MODEL}" | awk '{print $3, $4}')
    echo "  ✓ $MODEL ($SIZE)"
  else
    echo "  ✗ $MODEL — NOT FOUND"
    ALL_OK=false
  fi
done

echo ""

# ── Quick inference test ─────────────────────────────────────────────────────
echo "============================================"
echo " Running inference test..."
echo "============================================"
echo ""

for MODEL in "${MODELS[@]}"; do
  echo "[Test] $MODEL — simple prompt"
  echo "─────────────────────────────────────────────"

  RESPONSE=$(ollama run "$MODEL" "Reply with exactly: OK" 2>&1 | head -5)
  if [ -n "$RESPONSE" ]; then
    echo "  Response: $RESPONSE"
    echo "  ✓ PASS — $MODEL responds"
  else
    echo "  ✗ FAIL — no response from $MODEL"
    ALL_OK=false
  fi
  echo ""
done

# ── Cleanup ──────────────────────────────────────────────────────────────────
if [ "$STARTED_SERVER" = true ] && [ -n "$OLLAMA_PID" ]; then
  echo "Stopping ollama server (PID $OLLAMA_PID)..."
  kill "$OLLAMA_PID" 2>/dev/null || true
fi

# ── Summary ──────────────────────────────────────────────────────────────────
if [ "$ALL_OK" = true ]; then
  echo "============================================"
  echo " All models ready!"
  echo "============================================"
else
  echo "============================================"
  echo " Some models failed — check output above"
  echo "============================================"
  exit 1
fi

echo ""
echo "Usage:"
echo "  ollama run qwen2.5:3b \"Your prompt here\""
echo ""
echo "  # Or via API:"
echo "  curl http://localhost:11434/api/generate -d '{"
echo "    \"model\": \"qwen2.5:3b\","
echo "    \"prompt\": \"Your prompt here\""
echo "  }'"
echo ""
echo "  # RAG project uses this model automatically via:"
echo "  #   RAG_OLLAMA_MODEL=qwen2.5:3b (default in config/settings.py)"
echo ""
echo "Log saved to: $LOGFILE"
