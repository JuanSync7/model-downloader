#!/usr/bin/env bash
# BAAI Model Download & Update Script
# Downloads: BAAI/bge-m3 (embedding) · BAAI/bge-reranker-v2-m3 (reranker)
# Safe to re-run — checks latest commit on HuggingFace Hub and only
# re-downloads if the model has been updated since last run.
#
# Models saved to: ~/models/baai/
# Run with: bash ~/download-baai-models.sh

set -e
LOGFILE="$HOME/download-baai-models.log"
MODELS_DIR="$HOME/models/baai"
AI_ENV="$HOME/ai-env"
exec > >(tee -a "$LOGFILE") 2>&1

echo "============================================"
echo " BAAI Model Download / Update"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""

# ── Activate AI virtual environment ──────────────────────────────────────────
if [ ! -d "$AI_ENV" ]; then
  echo "ERROR: AI environment not found at $AI_ENV"
  echo "Run setup-ai.sh first: bash ~/setup-ai.sh"
  exit 1
fi

source "$AI_ENV/bin/activate"
echo "  ✓ AI environment activated"

# ── Install FlagEmbedding (official BAAI inference library) ──────────────────
echo ""
echo "Checking FlagEmbedding..."
FE_INSTALLED=$("$AI_ENV/bin/python" -m pip show FlagEmbedding 2>/dev/null | grep '^Version:' | awk '{print $2}')
FE_LATEST=$(curl -fsSL "https://pypi.org/pypi/FlagEmbedding/json" 2>/dev/null \
  | grep -o '"version":"[^"]*"' | head -1 | cut -d'"' -f4)

if [ -z "$FE_INSTALLED" ]; then
  echo "  → Installing FlagEmbedding $FE_LATEST..."
  uv pip install FlagEmbedding --quiet
  echo "  ✓ FlagEmbedding installed"
elif [ "$FE_INSTALLED" = "$FE_LATEST" ]; then
  echo "  ✓ FlagEmbedding $FE_INSTALLED (up to date)"
else
  echo "  ↑ FlagEmbedding $FE_INSTALLED → $FE_LATEST (upgrading)"
  uv pip install --upgrade FlagEmbedding --quiet
fi

mkdir -p "$MODELS_DIR"

# ── Python helpers (version check + download) ─────────────────────────────────
python - << PYEOF
import sys
from pathlib import Path
from huggingface_hub import model_info, snapshot_download

MODELS_DIR = Path("$MODELS_DIR")

def check_and_download(repo_id: str):
    """
    Downloads a model from HuggingFace Hub to MODELS_DIR/<model_name>.
    On re-runs, fetches the latest commit SHA from the Hub and skips
    the download if the local copy is already up to date.
    Ignores non-PyTorch weights (TF, Flax, ONNX) to save disk space.
    """
    model_name   = repo_id.split("/")[-1]
    local_path   = MODELS_DIR / model_name
    revision_file = local_path / ".last_revision"

    print(f"\n[{repo_id}]")

    # Fetch latest commit SHA from Hub
    try:
        info = model_info(repo_id)
        latest_sha = info.sha[:8]
    except Exception as e:
        print(f"  ✗ Could not reach HuggingFace Hub: {e}")
        sys.exit(1)

    # Compare with stored revision
    if local_path.exists() and revision_file.exists():
        stored_sha = revision_file.read_text().strip()
        if stored_sha == latest_sha:
            print(f"  ✓ Up to date (commit {latest_sha})")
            return str(local_path)
        else:
            print(f"  ↑ Update available: {stored_sha} → {latest_sha}")
    else:
        print(f"  → Downloading (commit {latest_sha})...")
        local_path.mkdir(parents=True, exist_ok=True)

    # Download — skip non-PyTorch formats to save space
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_path),
        local_dir_use_symlinks=False,
        ignore_patterns=[
            "*.msgpack",          # Flax/JAX weights
            "*.h5",               # TensorFlow weights
            "flax_model*",
            "tf_model*",
            "rust_model*",
            "onnx/*",             # ONNX exports
        ],
    )

    # Store the revision so we can check next time
    revision_file.write_text(latest_sha)
    print(f"  ✓ Saved to {local_path}")
    return str(local_path)


# ── Download models ───────────────────────────────────────────────────────────
print("\n" + "─" * 44)
print(" Step 1/2 — Embedding model")
print("─" * 44)
embed_path = check_and_download("BAAI/bge-m3")

print("\n" + "─" * 44)
print(" Step 2/2 — Reranker model")
print("─" * 44)
reranker_path = check_and_download("BAAI/bge-reranker-v2-m3")

print("\n✓ All models ready.")
PYEOF

# ── Run inference tests ───────────────────────────────────────────────────────
echo ""
echo "============================================"
echo " Running inference tests..."
echo "============================================"

python - << PYEOF
import sys
from pathlib import Path

MODELS_DIR = Path("$MODELS_DIR")
EMBED_PATH    = str(MODELS_DIR / "bge-m3")
RERANKER_PATH = str(MODELS_DIR / "bge-reranker-v2-m3")

# ── Test 1: BGE-M3 Embedding ─────────────────────────────────────────────────
print("\n[Test 1] BGE-M3 — dense embeddings + cosine similarity")
print("─" * 44)

try:
    from FlagEmbedding import BGEM3FlagModel
    import numpy as np

    model = BGEM3FlagModel(EMBED_PATH, use_fp16=True)

    sentences = [
        "Artificial intelligence is transforming industries.",
        "AI and machine learning are reshaping the world.",
        "The weather in Kuala Lumpur is hot and humid.",
    ]

    output = model.encode(
        sentences,
        batch_size=3,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    embeddings = output["dense_vecs"]

    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    sim_0_1 = cosine_sim(embeddings[0], embeddings[1])
    sim_0_2 = cosine_sim(embeddings[0], embeddings[2])

    print(f"  Embedding shape : {embeddings.shape}")
    print(f"  Similarity (AI sentences)      : {sim_0_1:.4f}  ← should be HIGH")
    print(f"  Similarity (AI vs weather)     : {sim_0_2:.4f}  ← should be LOW")

    if sim_0_1 > sim_0_2:
        print("  ✓ PASS — BGE-M3 embedding test")
    else:
        print("  ✗ FAIL — similarity scores look wrong")
        sys.exit(1)

except Exception as e:
    print(f"  ✗ BGE-M3 test failed: {e}")
    sys.exit(1)

# ── Test 2: BGE-Reranker-v2-M3 ───────────────────────────────────────────────
print("\n[Test 2] BGE-Reranker-v2-M3 — query/passage relevance scoring")
print("─" * 44)

try:
    from FlagEmbedding import FlagReranker

    reranker = FlagReranker(RERANKER_PATH, use_fp16=True)

    query = "What is machine learning?"
    relevant   = "Machine learning is a subset of AI that learns from data."
    irrelevant = "The Eiffel Tower is located in Paris, France."

    score_relevant   = reranker.compute_score([[query, relevant]],   normalize=True)
    score_irrelevant = reranker.compute_score([[query, irrelevant]], normalize=True)

    # compute_score returns a list for batch input; extract the single score
    if isinstance(score_relevant, list):
        score_relevant = score_relevant[0]
    if isinstance(score_irrelevant, list):
        score_irrelevant = score_irrelevant[0]

    print(f"  Relevant passage score   : {score_relevant:.4f}  ← should be HIGH")
    print(f"  Irrelevant passage score : {score_irrelevant:.4f}  ← should be LOW")

    if score_relevant > score_irrelevant:
        print("  ✓ PASS — BGE-Reranker-v2-M3 test")
    else:
        print("  ✗ FAIL — reranker scores look wrong")
        sys.exit(1)

except Exception as e:
    print(f"  ✗ Reranker test failed: {e}")
    sys.exit(1)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n============================================")
print(" All tests passed!")
print("============================================")
print(f"\nModels saved to: $MODELS_DIR")
print("""
Usage in Python:
  source ~/ai-env/bin/activate   # or: ai

  # Embedding
  from FlagEmbedding import BGEM3FlagModel
  embed = BGEM3FlagModel("$MODELS_DIR/bge-m3", use_fp16=True)
  vecs = embed.encode(["your text here"])["dense_vecs"]

  # Reranker
  from FlagEmbedding import FlagReranker
  reranker = FlagReranker("$MODELS_DIR/bge-reranker-v2-m3", use_fp16=True)
  score = reranker.compute_score([["query", "passage"]], normalize=True)
""")
PYEOF

echo "Log saved to: $LOGFILE"
