#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-35B-A3B-FP8}"
APP_DIR="${APP_DIR:-/opt/vllm-qwen}"
SERVICE_NAME="${SERVICE_NAME:-vllm-qwen}"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
FORCE_REINSTALL="${FORCE_REINSTALL:-0}"
AUTO_START="${AUTO_START:-0}"

echo "=================================================="
echo " vLLM + Qwen bootstrap"
echo " Model:        $MODEL_NAME"
echo " App dir:      $APP_DIR"
echo " Service:      $SERVICE_NAME"
echo " Port:         $PORT"
echo " Force update: $FORCE_REINSTALL"
echo " Auto start:   $AUTO_START"
echo "=================================================="

if [[ $EUID -ne 0 ]]; then
  echo "ERROR: Please run as root or with sudo."
  echo ""
  echo "Example:"
  echo "  sudo bash install.sh"
  exit 1
fi

log() {
  echo ""
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

service_exists() {
  systemctl list-unit-files "${SERVICE_NAME}.service" >/dev/null 2>&1
}

service_is_active() {
  systemctl is-active --quiet "$SERVICE_NAME" >/dev/null 2>&1
}

was_service_running=0

log "Checking GPU..."

if ! command_exists nvidia-smi; then
  echo "ERROR: nvidia-smi not found."
  echo "Use a GPU image with NVIDIA/CUDA driver already installed."
  exit 1
fi

nvidia-smi
GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"

if [[ "$GPU_COUNT" == "0" ]]; then
  echo "ERROR: No NVIDIA GPU detected."
  exit 1
fi

echo "Detected GPU count: $GPU_COUNT"

if service_exists && service_is_active; then
  was_service_running=1
  log "Existing $SERVICE_NAME service is running. Stopping it before update..."
  systemctl stop "$SERVICE_NAME"
fi

log "Installing required OS packages..."

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

git lfs install >/dev/null 2>&1 || true

log "Creating app folders..."

mkdir -p "$APP_DIR"
mkdir -p "$APP_DIR"/{cache,hf-cache,logs,scripts,models}
cd "$APP_DIR"

log "Installing/checking uv..."

export PATH="/root/.local/bin:$PATH"

if ! command_exists uv; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="/root/.local/bin:$PATH"
else
  echo "uv already installed: $(uv --version)"
fi

if ! command_exists uv; then
  echo "ERROR: uv install failed or uv not found in PATH."
  exit 1
fi

uv --version

log "Checking Python $PYTHON_VERSION..."

if ! uv python list | grep -q "cpython-${PYTHON_VERSION}"; then
  uv python install "$PYTHON_VERSION"
else
  echo "Python $PYTHON_VERSION already available in uv."
fi

log "Checking virtual environment..."

if [[ ! -f "$APP_DIR/.venv/bin/activate" ]]; then
  echo "Virtual environment not found. Creating new venv..."
  uv venv "$APP_DIR/.venv" --python "$PYTHON_VERSION" --seed
else
  echo "Virtual environment already exists: $APP_DIR/.venv"
fi

source "$APP_DIR/.venv/bin/activate"

log "Checking vLLM installation..."

vllm_installed=0

if python -c "import vllm; print(vllm.__version__)" >/tmp/vllm-version.txt 2>/dev/null; then
  vllm_installed=1
  echo "Existing vLLM version: $(cat /tmp/vllm-version.txt)"
fi

if [[ "$FORCE_REINSTALL" == "1" || "$vllm_installed" == "0" ]]; then
  echo "Installing/updating vLLM..."
  uv pip install -U pip
  uv pip install -U vllm --torch-backend=auto
  uv pip install -U huggingface_hub
else
  echo "vLLM already installed. Skipping reinstall."
  echo "To force update, run with:"
  echo "  FORCE_REINSTALL=1 bash install.sh"
fi

log "Writing environment file..."

cat > "$APP_DIR/env.sh" <<EOF
export MODEL_NAME="$MODEL_NAME"
export APP_DIR="$APP_DIR"

export HF_HOME="$APP_DIR/hf-cache"
export HUGGINGFACE_HUB_CACHE="$APP_DIR/hf-cache/hub"
export TRANSFORMERS_CACHE="$APP_DIR/hf-cache/transformers"
export VLLM_CACHE_ROOT="$APP_DIR/cache"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
EOF

chmod 644 "$APP_DIR/env.sh"

if [[ -n "${HF_TOKEN:-}" ]]; then
  log "HF_TOKEN provided. Logging in to Hugging Face..."
  source "$APP_DIR/env.sh"
  huggingface-cli login --token "$HF_TOKEN"
else
  echo "HF_TOKEN not provided. Skipping Hugging Face login."
fi

log "Writing start script..."

cat > "$APP_DIR/scripts/start-vllm.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/vllm-qwen}"

source "$APP_DIR/env.sh"
source "$APP_DIR/.venv/bin/activate"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
TP_SIZE="${TP_SIZE:-$(nvidia-smi -L | wc -l | tr -d ' ')}"

echo "Starting vLLM..."
echo "Model:        $MODEL_NAME"
echo "Host:         $HOST"
echo "Port:         $PORT"
echo "TP size:      $TP_SIZE"
echo "Max len:      $MAX_MODEL_LEN"
echo "Max seqs:     $MAX_NUM_SEQS"
echo "GPU util:     $GPU_MEMORY_UTILIZATION"

exec vllm serve "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --download-dir "$APP_DIR/models"
EOF

chmod +x "$APP_DIR/scripts/start-vllm.sh"

log "Writing systemd service..."

cat > "/etc/systemd/system/${SERVICE_NAME}.service" <<EOF
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
systemctl enable "$SERVICE_NAME"

if [[ "$AUTO_START" == "1" ]]; then
  log "AUTO_START=1, starting service..."
  systemctl restart "$SERVICE_NAME"
elif [[ "$was_service_running" == "1" ]]; then
  log "Service was running before update. Restarting it..."
  systemctl restart "$SERVICE_NAME"
else
  echo "Service is installed but not started."
fi

log "Final check..."

echo ""
echo "vLLM install path:"
echo "  $APP_DIR"
echo ""
echo "Start service:"
echo "  systemctl start $SERVICE_NAME"
echo ""
echo "Stop service:"
echo "  systemctl stop $SERVICE_NAME"
echo ""
echo "Restart service:"
echo "  systemctl restart $SERVICE_NAME"
echo ""
echo "View logs:"
echo "  journalctl -u $SERVICE_NAME -f"
echo ""
echo "Manual run:"
echo "  $APP_DIR/scripts/start-vllm.sh"
echo ""
echo "Test after startup:"
echo "  curl http://127.0.0.1:$PORT/v1/models | jq"
echo ""
echo "Force reinstall/update later:"
echo "  FORCE_REINSTALL=1 bash install.sh"
echo ""
echo "Run and auto-start immediately:"
echo "  AUTO_START=1 bash install.sh"
echo ""
echo "=================================================="
echo " Done"
echo "=================================================="