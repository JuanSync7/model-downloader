#!/usr/bin/env bash
# GLiNER Model Download & Update Script
# Downloads: urchade/gliner_medium-v2.1 (zero-shot NER)
# Safe to re-run — checks latest commit on HuggingFace Hub and only
# re-downloads if the model has been updated since last run.
#
# Model saved to: ~/models/gliner/
# Run with: bash ~/download-gliner-model.sh

set -e
LOGFILE="$HOME/download-gliner-model.log"
MODELS_DIR="$HOME/models/gliner"
AI_ENV="$HOME/ai-env"
exec > >(tee -a "$LOGFILE") 2>&1

echo "============================================"
echo " GLiNER Model Download / Update"
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

# ── Install gliner (official inference library) ──────────────────────────────
echo ""
echo "Checking gliner..."
GL_INSTALLED=$("$AI_ENV/bin/python" -m pip show gliner 2>/dev/null | grep '^Version:' | awk '{print $2}')
GL_LATEST=$(curl -fsSL "https://pypi.org/pypi/gliner/json" 2>/dev/null \
  | grep -o '"version":"[^"]*"' | head -1 | cut -d'"' -f4)

if [ -z "$GL_INSTALLED" ]; then
  echo "  → Installing gliner $GL_LATEST..."
  uv pip install gliner --quiet
  echo "  ✓ gliner installed"
elif [ "$GL_INSTALLED" = "$GL_LATEST" ]; then
  echo "  ✓ gliner $GL_INSTALLED (up to date)"
else
  echo "  ↑ gliner $GL_INSTALLED → $GL_LATEST (upgrading)"
  uv pip install --upgrade gliner --quiet
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


# ── Download model ───────────────────────────────────────────────────────────
print("\n" + "─" * 44)
print(" Step 1/1 — GLiNER model")
print("─" * 44)
gliner_path = check_and_download("urchade/gliner_medium-v2.1")

print("\n✓ Model ready.")
PYEOF

# ── Run inference test ───────────────────────────────────────────────────────
echo ""
echo "============================================"
echo " Running inference test..."
echo "============================================"

python - << PYEOF
import sys
from pathlib import Path

MODELS_DIR = Path("$MODELS_DIR")
GLINER_PATH = str(MODELS_DIR / "gliner_medium-v2.1")

# ── Test: GLiNER zero-shot NER ───────────────────────────────────────────────
print("\n[Test 1] GLiNER — zero-shot named entity recognition")
print("─" * 44)

try:
    from gliner import GLiNER

    model = GLiNER.from_pretrained(GLINER_PATH, local_files_only=True)

    text = (
        "Python is a programming language used with TensorFlow "
        "and PyTorch for building neural networks and transformers."
    )
    labels = [
        "technology", "algorithm", "framework", "concept",
        "programming language", "data structure",
    ]

    entities = model.predict_entities(text, labels, threshold=0.5)

    print(f"  Text   : {text[:70]}...")
    print(f"  Labels : {labels}")
    print(f"  Found  : {len(entities)} entities")
    for ent in entities:
        print(f"    • \"{ent['text']}\" → {ent['label']} ({ent['score']:.3f})")

    if len(entities) > 0:
        print("  ✓ PASS — GLiNER entity extraction test")
    else:
        print("  ✗ FAIL — no entities detected")
        sys.exit(1)

except Exception as e:
    print(f"  ✗ GLiNER test failed: {e}")
    sys.exit(1)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n============================================")
print(" All tests passed!")
print("============================================")
print(f"\nModel saved to: $MODELS_DIR")
print("""
Usage in Python:
  source ~/ai-env/bin/activate   # or: ai

  from gliner import GLiNER
  model = GLiNER.from_pretrained("$MODELS_DIR/gliner_medium-v2.1", local_files_only=True)
  entities = model.predict_entities(
      "Your text here",
      ["technology", "framework", "concept"],
      threshold=0.5,
  )
""")
PYEOF

echo "Log saved to: $LOGFILE"
