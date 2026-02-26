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

# ── Install / update ollama binary ───────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
  echo "  Ollama not found — installing..."
  curl -fsSL https://ollama.com/install.sh | sh
  echo "  ✓ Ollama installed"
else
  OLLAMA_VERSION=$(ollama --version 2>&1 | awk '{print $NF}')
  echo "  Ollama $OLLAMA_VERSION already installed"
  echo "  Checking for updates..."
  curl -fsSL https://ollama.com/install.sh | sh
  NEW_VERSION=$(ollama --version 2>&1 | awk '{print $NF}')
  if [ "$OLLAMA_VERSION" = "$NEW_VERSION" ]; then
    echo "  ✓ Ollama $OLLAMA_VERSION (up to date)"
  else
    echo "  ↑ Ollama $OLLAMA_VERSION → $NEW_VERSION (upgraded)"
  fi
fi
echo ""

# ── Ensure ollama server is running ──────────────────────────────────────────
if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
  echo "  Starting ollama server..."
  ollama serve &>/dev/null &
  OLLAMA_PID=$!
  STARTED_SERVER=true
  # Wait for server to be ready
  for i in {1..10}; do
    if curl -sf http://localhost:11434/api/tags &>/dev/null; then
      break
    fi
    sleep 1
  done
  if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
    echo "  ERROR: Could not start ollama server"
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
