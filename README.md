# llama-cpp-turboquant — Docker Setup

Servidor de inferência local baseado no fork [llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant), com quantização TurboQuant+ de KV cache e suporte a NVIDIA CUDA.

---

## Referências do Projeto

| Recurso | Link |
|---|---|
| Fork llama-cpp-turboquant | https://github.com/TheTom/llama-cpp-turboquant |
| Papers TurboQuant+ | https://github.com/TheTom/turboquant_plus/tree/main/docs/papers |
| Modelo Granite 4.1 | https://ollama.com/library/granite4.1 |
| Arquitetura Granite 4.1 | https://huggingface.co/blog/ibm-granite/granite-4-1 |
| llama.cpp upstream | https://github.com/ggml-org/llama.cpp |

---

## Hardware Alvo

| Componente | Especificação |
|---|---|
| GPU | NVIDIA RTX 3060 — 12GB VRAM |
| CPU | AMD Ryzen 5 7600X — 6 cores físicos / 12 threads |
| RAM | 32GB DDR5 |

---

## Modelo

**IBM Granite 4.1 30B Instruct** — quantização `Q4_K_M`

| Propriedade | Valor |
|---|---|
| Tamanho em disco (Q4_K_M) | ~17 GB |
| Camadas | 64 |
| KV heads | 8 |
| Contexto máximo | 512K tokens |
| Arquitetura | Denso, decoder-only |
| Licença | Apache 2.0 |

---

## Configuração das Flags

### `--ngl 38` — GPU Layers

Define quantas camadas do modelo são carregadas na VRAM da GPU.

**Cálculo:**

```
Q4_K_M 30B ≈ 17 GB / 64 camadas = ~265 MB por camada
12 GB VRAM − 1 GB overhead − ~500 MB KV cache = ~10.5 GB disponíveis
10.5 GB / 265 MB ≈ 39 camadas → conservador: 38
```

**Resultado:** 38 camadas na GPU (~10 GB) + 26 camadas na CPU/RAM (~7 GB). Hybrid inference estável sem risco de OOM.

---

### `--n-cpu-moe 6` — CPU Threads para MoE

Define a quantidade de threads da CPU usadas para processar experts em modelos Mixture-of-Experts (MoE).

O Granite 4.1 30B é um modelo **denso** (não MoE), portanto esta flag não tem efeito neste caso específico. É mantida como boa prática para compatibilidade com eventuais modelos MoE futuros (DeepSeek, Mixtral, Qwen MoE).

O Ryzen 7600X possui **6 cores físicos**, valor usado aqui.

---

### `--no-mmap` — Sem Memory-Map

Desativa o mapeamento de memória (mmap) e carrega o modelo inteiro na RAM antes de iniciar.

**Motivo:** Com 32 GB de RAM, os ~17 GB do modelo cabem integralmente. Sem mmap, o acesso é direto e contínuo, eliminando latência de page faults durante inferência — especialmente relevante em sessões longas ou com contexto grande.

---

### `--cache-type-k q8_0` — Quantização do K Cache

Mantém o cache de **Keys** (K) em 8-bit.

**Motivo:** De acordo com o paper [Asymmetric K/V Cache Compression](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md) do repositório:

> *"K is everything — V is free. Compressing K aggressively causes PPL blow-up on certain model families."*

K deve sempre ficar em precisão maior que V. `q8_0` é o mínimo recomendado para K em produção.

---

### `--cache-type-v turbo3` — Quantização do V Cache

Comprime o cache de **Values** (V) com o codec TurboQuant `turbo3` (~3.5 bits por elemento).

**Escala de compressão V (do menos para o mais agressivo):**

| Tipo | Bits/elem | Compressão | Qualidade |
|---|---|---|---|
| `turbo4` | ~4.5 | menor | melhor (conservador) |
| **`turbo3`** | **~3.5** | **~4.6×** | **<1% PPL loss** ✅ |
| `turbo2` | ~2.0 | maior | >2% PPL loss |

> ⚠️ O número maior indica **mais bits**, não mais compressão. `turbo3` comprime mais que `turbo4`.

**Motivo da escolha:** `q8_0/turbo3` é o *"recommended default for most production workloads"* conforme o repositório. Com o Granite 4.1 30B (modelo bem comportado e denso), não há necessidade da abordagem mais conservadora (`turbo4`). O KV cache comprimido libera VRAM adicional para acomodar o contexto longo de até 512K tokens.

---

## Resumo Visual

```
┌─────────────────────────────────────────────────────────┐
│  RTX 3060 12GB VRAM                                      │
│  ┌──────────────────────────────────────────────────┐   │
│  │ 38 camadas (layers 0–37) ≈ 10 GB                 │   │
│  │ KV cache (turbo3 V + q8_0 K) ≈ 500 MB           │   │
│  │ overhead CUDA ≈ 1 GB                             │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  RAM 32 GB                                               │
│  ┌──────────────────────────────────────────────────┐   │
│  │ 26 camadas (layers 38–63) ≈ 7 GB                 │   │
│  │ modelo carregado via --no-mmap ≈ 17 GB total     │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Como Usar

### Pré-requisitos

- Docker + Docker Compose
- [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) instalado
- Arquivo GGUF do modelo em um volume Docker acessível

### Build e Start

```bash
docker compose up --build
```

### Substituir o Modelo

Edite a flag `-m` no `command` do `docker-compose.yml`:

```yaml
command: >
  -m /models/SEU_MODELO.gguf
  ...
```

Certifique-se de ajustar `--n-gpu-layers` conforme o tamanho e número de camadas do novo modelo.

---

### Acessar a API

O servidor expõe uma API compatível com OpenAI em:

```
http://localhost:8080
```

Exemplo:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite4.1-30b",
    "messages": [{"role": "user", "content": "Olá!"}]
  }'
```
