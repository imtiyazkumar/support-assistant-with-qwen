# qwen-lora

**A local Yember customer and partner support assistant, powered by Qwen3.5-2B, FAQ retrieval, and an optional QLoRA adapter.**

This project runs a support chat from the terminal. It searches a curated FAQ in `docs/train.jsonl`, supplies the matching entry to a locally hosted language model, and displays the source question used. The FAQ is the source of Yember-specific facts; fine-tuning adapts the model's response behavior but does not replace retrieval.

The current implementation is an experimental local prototype. It is not a hosted service, and its generated answers still need review before production use.

## Contents

- [Features](#features)
- [How answers are produced](#how-answers-are-produced)
- [Requirements](#requirements)
- [Run the assistant](#run-the-assistant)
- [Validate and evaluate](#validate-and-evaluate)
- [Update the FAQ and train the adapter](#update-the-faq-and-train-the-adapter)
- [Configuration and privacy](#configuration-and-privacy)
- [Project layout](#project-layout)
- [Current limitations](#current-limitations)

## Features

- **Local inference:** runs the pinned Qwen3.5-2B text model from local files on a CUDA GPU.
- **FAQ retrieval:** searches the questions and answers in `docs/train.jsonl` with a small weighted keyword retriever. It handles some common wording variations and a few known typos.
- **Grounded prompt:** provides one matching FAQ entry to the model and shows the FAQ question after the answer.
- **Safe fallback:** returns a support-contact response when no FAQ entry sufficiently matches the query.
- **Optional QLoRA fine-tuning:** trains a small adapter using the same FAQ data. If `models/adapters/yember-support` exists, the runtime loads it automatically.
- **Offline runtime:** after dependencies and model weights are installed, chat and verification use local files. Runtime caches and telemetry are configured locally.

## How answers are produced

```text
Question
   │
   ▼
FAQ keyword retrieval ── no suitable match ──► Safe fallback
   │ match
   ▼
FAQ question and answer + chat instructions
   │
   ▼
Local Qwen3.5-2B (+ optional Yember LoRA adapter)
   │
   ▼
Generated answer and the matched FAQ question
```

Retrieval and fine-tuning have separate jobs:

| Component | Purpose | Current implementation |
| --- | --- | --- |
| FAQ retrieval (RAG) | Supplies up-to-date Yember facts at answer time | Weighted keyword overlap over `docs/train.jsonl`; no embeddings or vector database |
| QLoRA fine-tuning | Teaches the model response style and task behavior | Optional adapter trained from the FAQ examples |
| Base model | Generates a conversational response | Local Qwen3.5-2B text model |

The retriever does not guarantee that every paraphrase will match, and the language model can still misstate or omit details from a retrieved answer. Always compare important operational answers with the cited FAQ entry and current application behavior.

## Requirements

The verified setup is Ubuntu 26.04 under WSL2, Python 3.12.16, and an NVIDIA RTX 3050 Laptop GPU with 6 GB VRAM. The configured runtime uses BF16 on CUDA and requires a BF16-capable GPU. Other systems may work but have not been verified. The pinned native acceleration wheel targets Linux x86-64 and Python 3.12.

You will also need Git and [`uv`](https://docs.astral.sh/uv/). The model weights are approximately 4.3 GB. Dependencies, caches, base weights, and trained adapters require additional disk space.

On WSL2, GPU access comes from the Windows NVIDIA driver. Do not install a Linux display driver inside WSL. A CUDA toolkit is not required for the prebuilt packages used here.

## Run the assistant

### 1. Prepare the environment

From an Ubuntu shell, clone the repository and create the Python environment:

```bash
git clone https://github.com/imtiyazkumar/qwen-lora.git
cd qwen-lora

uv --no-config --cache-dir .cache/uv python install 3.12.16
uv --no-config --cache-dir .cache/uv venv --python 3.12.16 .venv
```

Install the tested CUDA-enabled dependencies:

```bash
uv --no-config --cache-dir .cache/uv pip install \
  --python .venv/bin/python \
  --default-index https://pypi.org/simple \
  --torch-backend cu128 \
  --only-binary :all: \
  -r config/requirements-training.lock.txt

uv --no-config --cache-dir .cache/uv pip check --python .venv/bin/python
```

The project lock includes PyTorch with CUDA 12.8, Transformers, Unsloth, PEFT, TRL, and the matching native `causal-conv1d` package. The first installation requires internet access.

### 2. Download the base model

```bash
.venv/bin/python scripts/download_model.py
```

The downloader fetches the model revision pinned in `config/model.json`, checks the available file hashes, and stores weights under `models/base/Qwen3.5-2B`. This step requires internet access. Chat runs offline from these local files afterward.

### 3. Check the FAQ and launch chat

```bash
source scripts/local_env.sh
.venv/bin/python scripts/verify_yember_data.py
.venv/bin/python scripts/chat_local.py
```

Type a question at the `You:` prompt. Use `/clear` to reset conversation history or `/quit` to exit. Chat history lasts only for that process. The configured limits are 768 input tokens and 128 generated tokens.

Try questions directly represented in the FAQ, then try paraphrases and unsupported questions. For example:

```text
How do customers sign in?
How can I onboard a restaurant?
What happens after a delivery partner pays the registration fee?
What is the status of my application?
```

The assistant prints a `[Yember FAQ: ...]` line for a retrieved answer. Compare the generated answer with that FAQ entry: the source line confirms what was retrieved, not that the model's response is fully correct.

## Validate and evaluate

### FAQ structure

Run the validator:

```bash
.venv/bin/python scripts/verify_yember_data.py
```

It checks JSONL format, required message roles, nonempty content, duplicate questions, consistent system instructions, answer length, and likely accidental secrets. It does **not** validate product facts against the live Yember applications.

### Retrieval coverage

The following check asks the retriever every dataset question and verifies that it selects that question's own FAQ entry:

```bash
source scripts/local_env.sh
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from backend.services.yember_knowledge import YemberKnowledgeBase

knowledge = YemberKnowledgeBase()
rows = [
    json.loads(line)["messages"]
    for line in Path("docs/train.jsonl").read_text().splitlines()
]
matched = sum(
    knowledge.answer_context(row[1]["content"])[1] == row[1]["content"]
    for row in rows
)
print(f"Exact FAQ retrieval: {matched}/{len(rows)}")
PY
```

This checks exact FAQ questions against retrieval. It does not measure paraphrase coverage or the correctness of generated responses.

### Model inference checks

Run `scripts/verify_model.py` to verify the local model and its inference smoke prompts:

```bash
source scripts/local_env.sh
.venv/bin/python scripts/verify_model.py
```

For support-answer checks, use `scripts/chat_local.py` and compare each generated response with its printed FAQ source. There is not yet an automated evaluator that checks the factual correctness of generated answers across all FAQ records.

## Update the FAQ and train the adapter

Each line of `docs/train.jsonl` is one JSON object with three messages: `system`, `user`, and `assistant`. Keep answers grounded in verified Yember behavior, distinguish customer, restaurant, delivery-partner, and administrator actions, and state when the source material does not specify a detail.

After editing the FAQ, validate it and run a training-only data/model check:

```bash
source scripts/local_env.sh
.venv/bin/python scripts/verify_yember_data.py
.venv/bin/python scripts/train_yember.py --validate-only
```

When the data is ready, train the adapter on a BF16-capable CUDA GPU:

```bash
.venv/bin/python scripts/train_yember.py
```

The default training setup uses a deterministic 90/10 split, 4-bit NF4 base weights, rank-8 LoRA, a 512-token sequence limit, and three epochs. Training needs substantial free GPU memory and several minutes. Close chat and other GPU processes first. The adapter is saved under `models/adapters/yember-support`; restart chat after training to load it.

Fine-tuning does not add new facts to retrieval or guarantee faithful answers. The FAQ remains the runtime source for Yember-specific information.

## Configuration and privacy

- Model, revision, precision, device, and token budgets are configured in `config/model.json`.
- `scripts/local_env.sh` places Hugging Face and compilation caches inside `.cache/` and defaults to offline operation.
- The base model, adapters, virtual environment, caches, training checkpoints, and local document stores are excluded by `.gitignore`.
- Initial dependency and model downloads use the network. Runtime chat is configured for local files; verification scripts' Python socket checks are not system-wide network isolation.
- Review `docs/train.jsonl` before sharing it. Do not put customer personal information, credentials, or private operational data in the training set.

## Project layout

```text
backend/
  inference/runtime.py          Local model loading, adapter loading, and generation
  services/yember_knowledge.py  Keyword retrieval over the FAQ
  services/environment.py       Offline defaults and local cache configuration
config/
  model.json                    Model revision and runtime limits
  requirements-*.txt            Tested dependency lists and locks
docs/
  train.jsonl                   Yember support and training examples
  setup.md                      Detailed environment setup notes
  model-selection.md            Model and runtime decisions
  learning-guide.md              Concepts and implementation guide
evaluation/
  datasets/                      Model smoke-test prompts
  reports/                       Saved GPU, inference, and training reports
scripts/
  chat_local.py                  Interactive local support chat
  train_yember.py                Optional QLoRA training command
  verify_yember_data.py          FAQ structure and privacy checks
  verify_model.py                Local model smoke tests and benchmarks
  download_model.py              Pinned base model download and verification
  local_env.sh                   Local cache and offline environment settings
```

## Current limitations

- The knowledge source is a curated FAQ. PDF/DOCX ingestion, embeddings, a vector database, and a web interface are not implemented.
- Retrieval is keyword-based and can miss differently worded questions or select an imperfect entry.
- Generated answers are not guaranteed to preserve every condition or actor in the source. A recent delivery-partner smoke check incorrectly implied that payment and document submission lead to approval; the FAQ says administrator approval is separate. Review generated answers before using this for live support.
- The saved fine-tuning report uses 15 validation examples. Its loss and smoke checks do not establish broad support accuracy.
- The FAQ's product statements must be checked against the current Yember application and operational policy.

See [docs/setup.md](docs/setup.md) for expanded installation and GPU notes, [docs/learning-guide.md](docs/learning-guide.md) for an implementation walkthrough, and [evaluation/reports/yember-training.json](evaluation/reports/yember-training.json) for recorded adapter training metrics.
