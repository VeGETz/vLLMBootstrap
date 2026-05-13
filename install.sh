#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-35B-A3B}"
APP_DIR="${APP_DIR:-/opt/vllm-qwen}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"

echo "=================================================="
echo " vLLM + Qwen bootstrap"
echo " Model: $MODEL_NAME"
echo " App dir: $APP_DIR"
echo " Port: $PORT"
echo "=================================================="

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root or with sudo:"
  echo "sudo bash install-vllm-qwen.sh"
  exit 1
fi

echo "[1/9] Checking GPU..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found."
  echo "Use a Spheron image with NVIDIA/CUDA driver already installed."
  exit 1
fi

nvidia-smi
GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
echo "Detected GPU count: $GPU_COUNT"

echo "[2/9] Installing OS packages..."
apt update
DEBIAN_FRONTEND=noninteractive apt install -y \
  curl \
  wget \
  git \
  git-lfs \
  htop \
  tmux \
  jq \
  build-essential \
  python3 \
  python3-pip \
  python3-venv \
  ca-certificates

git lfs install

echo "[3/9] Creating app folders..."
mkdir -p "$APP_DIR"
mkdir -p "$APP_DIR"/{cache,hf-cache,logs,scripts}
cd "$APP_DIR"

echo "[4/9] Installing uv..."
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="/root/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv install failed or not in PATH."
  exit 1
fi

uv --version

echo "[5/9] Creating Python 3.12 venv..."
uv python install 3.12
uv venv "$APP_DIR/.venv" --python 3.12 --seed

source "$APP_DIR/.venv/bin/activate"

echo "[6/9] Installing vLLM..."
uv pip install -U pip
uv pip install -U vllm --torch-backend=auto
uv pip install -U huggingface_hub

echo "[7/9] Configuring environment..."
cat > "$APP_DIR/env.sh" <<EOF
export MODEL_NAME="$MODEL_NAME"
export APP_DIR="$APP_DIR"
export HF_HOME="$APP_DIR/hf-cache"
export HUGGINGFACE_HUB_CACHE="$APP_DIR/hf-cache/hub"
export TRANSFORMERS_CACHE="$APP_DIR/hf-cache/transformers"
export VLLM_CACHE_ROOT="$APP_DIR/cache"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
EOF

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN provided, logging in to Hugging Face..."
  source "$APP_DIR/env.sh"
  huggingface-cli login --token "$HF_TOKEN"
else
  echo "HF_TOKEN not provided. Skipping Hugging Face login."
  echo "If model download fails later, rerun with HF_TOKEN=hf_xxx before the curl command."
fi

echo "[8/9] Creating start script..."
cat > "$APP_DIR/scripts/start-vllm.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/vllm-qwen}"
source "$APP_DIR/env.sh"
source "$APP_DIR/.venv/bin/activate"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
TP_SIZE="${TP_SIZE:-$(nvidia-smi -L | wc -l | tr -d ' ')}"

exec vllm serve "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --download-dir "$APP_DIR/models"
EOF

chmod +x "$APP_DIR/scripts/start-vllm.sh"

echo "[9/9] Creating systemd service..."
cat > /etc/systemd/system/vllm-qwen.service <<EOF
[Unit]
Description=vLLM Qwen Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=$APP_DIR
Environment=APP_DIR=$APP_DIR
Environment=PORT=$PORT
Environment=HOST=$HOST
Environment=MAX_MODEL_LEN=$MAX_MODEL_LEN
Environment=GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
Environment=MAX_NUM_SEQS=$MAX_NUM_SEQS
ExecStart=$APP_DIR/scripts/start-vllm.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable vllm-qwen

echo "=================================================="
echo " Installation finished."
echo ""
echo " Start server:"
echo "   systemctl start vllm-qwen"
echo ""
echo " View logs:"
echo "   journalctl -u vllm-qwen -f"
echo ""
echo " Test after startup:"
echo "   curl http://127.0.0.1:$PORT/v1/models | jq"
echo ""
echo " Manual run:"
echo "   $APP_DIR/scripts/start-vllm.sh"
echo "=================================================="