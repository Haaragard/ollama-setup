#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
err()   { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; exit 1; }

require() {
  command -v "$1" &>/dev/null || err "'$1' não encontrado. Instale e tente novamente."
}

usage() {
  cat <<EOF
Uso: $0 -r <hf_repo> -q <quantização> -o <arquivo_saída> [-v <volume>] [-w <work_dir>]

Parâmetros obrigatórios:
  -r  Repositório HuggingFace         Ex: ibm-granite/granite-4.1-30b
  -q  Algoritmo de quantização        Ex: Q4_K_M, Q5_K_M, Q8_0, F16
  -o  Nome do arquivo GGUF de saída   Ex: granite4.1-30b-Q4_K_M.gguf

Parâmetros opcionais:
  -v  Nome do volume Docker destino   (padrão: ollama_llama_models)
  -w  Diretório de trabalho local     (padrão: ./work)
  -h  Exibe esta ajuda

Variáveis de ambiente:
  HF_TOKEN  Token do HuggingFace para modelos privados/gated

Exemplos:
  $0 -r ibm-granite/granite-4.1-30b -q Q4_K_M -o granite4.1-30b-Q4_K_M.gguf
  $0 -r meta-llama/Llama-3-8b -q Q5_K_M -o llama3-8b-Q5_K_M.gguf -v meu_volume
EOF
  exit 0
}

# ─────────────────────────────────────────────
# Parsing de parâmetros
# ─────────────────────────────────────────────
HF_REPO=""
QUANT=""
OUTPUT_GGUF=""
VOLUME_NAME="ollama_llama_models"
WORK_DIR="$(pwd)/work"

while getopts "r:q:o:v:w:h" opt; do
  case $opt in
    r) HF_REPO="$OPTARG" ;;
    q) QUANT="$OPTARG" ;;
    o) OUTPUT_GGUF="$OPTARG" ;;
    v) VOLUME_NAME="$OPTARG" ;;
    w) WORK_DIR="$OPTARG" ;;
    h) usage ;;
    *) usage ;;
  esac
done

[[ -z "$HF_REPO"     ]] && err "Parâmetro -r (repositório HuggingFace) é obrigatório."
[[ -z "$QUANT"       ]] && err "Parâmetro -q (algoritmo de quantização) é obrigatório."
[[ -z "$OUTPUT_GGUF" ]] && err "Parâmetro -o (arquivo de saída) é obrigatório."

# Garante extensão .gguf
[[ "$OUTPUT_GGUF" != *.gguf ]] && OUTPUT_GGUF="${OUTPUT_GGUF}.gguf"

# Nome base para o F16 intermediário
MODEL_NAME="${OUTPUT_GGUF%.gguf}"

# ─────────────────────────────────────────────
# Config derivada
# ─────────────────────────────────────────────
HF_CACHE="${WORK_DIR}/hf_model"
GGUF_DIR="${WORK_DIR}/gguf"

# ─────────────────────────────────────────────
# Verificações iniciais
# ─────────────────────────────────────────────
require docker
require python3

mkdir -p "${HF_CACHE}" "${GGUF_DIR}"

# ─────────────────────────────────────────────
# 1. Instalar dependências Python em virtualenv
# ─────────────────────────────────────────────
VENV_DIR="${WORK_DIR}/.venv"
info "Verificando dependências Python..."
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi
"${VENV_DIR}/bin/pip" install -q --upgrade huggingface_hub

# ─────────────────────────────────────────────
# 2. Download do modelo (safetensors)
# ─────────────────────────────────────────────
info "Baixando modelo '${HF_REPO}' do HuggingFace..."
info "Isso pode demorar bastante (~60GB). Use HF_TOKEN para modelos privados."

"${VENV_DIR}/bin/python3" - <<PYEOF
from huggingface_hub import snapshot_download
import os

token = os.environ.get("HF_TOKEN", None)

snapshot_download(
    repo_id="${HF_REPO}",
    local_dir="${HF_CACHE}",
    ignore_patterns=["*.md", "*.sig", ".gitattributes"],
    token=token,
)
print("Download concluído.")
PYEOF

ok "Modelo baixado em: ${HF_CACHE}"

# ─────────────────────────────────────────────
# 3. Conversão para GGUF usando llama.cpp (Docker)
#    Imagem: ghcr.io/ggml-org/llama.cpp:full
# ─────────────────────────────────────────────

# Se a quantização pedida for F16, geramos o arquivo final diretamente
if [[ "$QUANT" == "F16" ]]; then
  info "Convertendo safetensors → GGUF F16 (arquivo final)..."

  docker run --rm \
    -v "${HF_CACHE}:/input:ro" \
    -v "${GGUF_DIR}:/output" \
    ghcr.io/ggml-org/llama.cpp:full \
    python3 /app/convert_hf_to_gguf.py \
      /input \
      --outtype f16 \
      --outfile "/output/${OUTPUT_GGUF}"

  ok "GGUF gerado: ${GGUF_DIR}/${OUTPUT_GGUF}"
else
  # Gera F16 intermediário e depois quantiza
  info "Convertendo safetensors → GGUF F16 intermediário..."

  docker run --rm \
    -v "${HF_CACHE}:/input:ro" \
    -v "${GGUF_DIR}:/output" \
    ghcr.io/ggml-org/llama.cpp:full \
    python3 /app/convert_hf_to_gguf.py \
      /input \
      --outtype f16 \
      --outfile "/output/${MODEL_NAME}-f16.gguf"

  ok "GGUF F16 intermediário: ${GGUF_DIR}/${MODEL_NAME}-f16.gguf"

  # ─────────────────────────────────────────────
  # 4. Quantização
  # ─────────────────────────────────────────────
  info "Quantizando para ${QUANT}..."

  docker run --rm \
    -v "${GGUF_DIR}:/models" \
    ghcr.io/ggml-org/llama.cpp:full \
    /app/llama-quantize \
      "/models/${MODEL_NAME}-f16.gguf" \
      "/models/${OUTPUT_GGUF}" \
      "${QUANT}"

  ok "Quantização concluída: ${GGUF_DIR}/${OUTPUT_GGUF}"

  info "Removendo arquivo F16 intermediário..."
  rm -f "${GGUF_DIR}/${MODEL_NAME}-f16.gguf"
fi

# ─────────────────────────────────────────────
# 5. Copiar para o Docker volume
# ─────────────────────────────────────────────
info "Copiando ${OUTPUT_GGUF} para o volume Docker '${VOLUME_NAME}'..."

docker run --rm \
  -v "${GGUF_DIR}:/src:ro" \
  -v "${VOLUME_NAME}:/models" \
  alpine \
  cp "/src/${OUTPUT_GGUF}" "/models/${OUTPUT_GGUF}"

ok "Modelo disponível em /models/${OUTPUT_GGUF} no volume '${VOLUME_NAME}'"

# ─────────────────────────────────────────────
# 6. Resumo
# ─────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║           Concluído com sucesso!         ║"
echo "╠══════════════════════════════════════════╣"
printf  "║  Modelo: %-32s║\n" "${OUTPUT_GGUF}"
printf  "║  Volume: %-32s║\n" "${VOLUME_NAME}"
echo "╚══════════════════════════════════════════╝"
echo ""
info "Agora suba o serviço com: docker compose up -d"
