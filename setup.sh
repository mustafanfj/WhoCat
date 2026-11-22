#!/usr/bin/env bash
# One-time setup. Run from the repo folder (cloned as <project>/code):  bash setup.sh
set -e
CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$CODE_DIR/.."
PROJECT="$(pwd)"
[ -d .venv ] || python3 -m venv --system-site-packages .venv
source .venv/bin/activate
cat > env.sh <<EOT
export PROJECT_ROOT="$PROJECT"
export WHOCAT_ROOT="$PROJECT"
export XDG_CACHE_HOME="$PROJECT/.cache"
export HF_HOME="$PROJECT/.cache/huggingface"
export TORCH_HOME="$PROJECT/.cache/torch"
export PIP_CACHE_DIR="$PROJECT/.cache/pip"
source "$PROJECT/.venv/bin/activate"
# venv packages must win over older ones in ~/.local (user site loads before the venv otherwise)
export PYTHONPATH="$PROJECT/.venv/lib/python3.10/site-packages\${PYTHONPATH:+:\$PYTHONPATH}"
EOT
source env.sh
python -m pip install -q --upgrade pip
python -m pip install -q -r "$CODE_DIR/requirements.txt"
python - <<'PY'
import torch, timm, transformers
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(), "| GPUs", torch.cuda.device_count())
print("timm", timm.__version__, "| transformers", transformers.__version__)
PY
echo "Setup done. In every new shell:  source \"$PROJECT/env.sh\""
