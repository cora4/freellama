# freellama.cpp

**Fast, local inference for open-weight language models.**

Freellama is a lightweight inference runtime and API server for running large language models locally or on dedicated GPU infrastructure.

It provides a simple interface for loading a model, generating text, streaming tokens, and exposing an OpenAI-compatible HTTP API.

## Features

* 🚀 Fast inference
* 💻 Local CPU and GPU execution
* 🧠 Support for large language models
* 🔄 Streaming token generation
* 🌐 OpenAI-compatible API
* 📦 Quantised model support
* 📊 Token and performance metrics
* 🔌 Python and HTTP interfaces
* 🔒 Models and prompts can remain on your infrastructure

## Architecture

```text
                    ┌─────────────────────┐
                    │       Client        │
                    │                     │
                    │  Python / curl / UI │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      HTTP API       │
                    │                     │
                    │ /v1/chat/completions|
                    │  /v1/completions    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Inference Engine │
                    │                     │
                    │ Tokenization        │
                    │ KV Cache            │
                    │ Batching            │
                    │ Sampling            │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
          ┌─────────────┐             ┌─────────────┐
          │     CPU     │             │     GPU     │
          │             │             │ CUDA/Metal  │
          └─────────────┘             └─────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       Model         │
                    │                     │
                    │  Model weights      │
                    │  Tokeniser          │
                    │  Configuration      │
                    └─────────────────────┘
```

## Requirements

### Hardware

The required hardware depends on the model being served.

For example:

| Component | Minimum |
| --- | --- |
| CPU | Modern 64-bit CPU |
| RAM | 16 GB+ |
| GPU | Optional |
| GPU memory | Model dependent |
| Storage | Model-size dependent |

GPU acceleration is recommended for production workloads.

### Software

* Python 3.10+
* Git
* CUDA-compatible GPU (optional)
* CUDA Toolkit (if using NVIDIA acceleration)

## Installation

Clone the repository:

```bash
git clone https://github.com/example/Freellama.git
cd Freellama
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the package:

```bash
pip install -e .
```

## Quick Start

Download a compatible model and start the server:

```bash
Freellama serve ./models/my-model
```

The server will start on:

```text
http://localhost:8000
```

Check the server:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok"
}
```

## Generate Text

Send a completion request:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "prompt": "Explain quantum computing in simple terms.",
    "max_tokens": 200
  }'
```

## Chat Completions

Freellama provides an OpenAI-compatible chat endpoint:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [
      {
        "role": "user",
        "content": "What is inference in AI?"
      }
    ],
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

## Streaming

Set `stream` to `true` to receive tokens as they are generated:

```json
{
  "model": "my-model",
  "messages": [
    {
      "role": "user",
      "content": "Write a short poem about space."
    }
  ],
  "stream": true
}
```

The server returns Server-Sent Events (SSE):

```text
data: {"choices":[{"delta":{"content":"Beyond"}}]}

data: {"choices":[{"delta":{"content":" the"}}]}

data: {"choices":[{"delta":{"content":" stars"}}]}
```

## Python

You can also use the HTTP API from Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "my-model",
        "messages": [
            {
                "role": "user",
                "content": "Hello!"
            }
        ],
        "max_tokens": 100,
    },
)

print(response.json())
```

## Configuration

Freellama can be configured using command-line arguments or environment variables.

Example:

```bash
freellama serve ./models/my-model \
  --host 0.0.0.0 \
  --port 8000 \
  --device cuda \
  --dtype bf16 \
  --max-context 8192 \
  --max-batch-size 8
```

### Environment Variables

```bash
FREELLAMA_HOST=0.0.0.0
FREELLAMA_PORT=8000
FREELLAMA_DEVICE=cuda
FREELLAMA_DTYPE=bf16
FREELLAMA_MAX_CONTEXT=8192
FREELLAMA_MAX_BATCH_SIZE=8
```

## Quantization

Quantization reduces the memory required to run a model.

Supported formats can include:

* FP32
* FP16
* BF16
* INT8
* INT4
* MXFP4

Example:

```bash
Freellama serve ./models/my-model \
  --quantization int4
```

The exact supported formats depend on the model architecture and backend.

## GPU Backends

Freellama is designed around a backend abstraction:

```text
                 Inference Engine
                        │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
             GPU      OTHER       CPU
              │         │         │
           NVIDIA     OTHER     x86/ARM
```

This allows hardware-specific implementations without changing the API exposed to applications.

## Performance

Inference performance depends on:

* Model size
* Quantization
* Context length
* Batch size
* GPU architecture
* Available memory bandwidth
* Number of concurrent requests
* Sampling configuration

Freellama exposes basic metrics such as:

```text
Prompt tokens
Generated tokens
Time to first token
Tokens per second
Total latency
Queue latency
GPU memory usage
```

Example:

```text
Model:             my-model
Prompt tokens:     128
Generated tokens:  256
Time to first token: 92 ms
Generation speed:  78 tok/s
Total latency:     3.37 s
```

## Continuous Batching

For server workloads, Freellama can combine multiple requests into a shared inference step.

```text
Request A ──┐
Request B ──┼──► Scheduler ──► GPU
Request C ──┘
```

This can improve GPU utilization when multiple users are generating text simultaneously.

## KV Cache

During autoregressive generation, previously computed attention states can be stored in a **Key-Value cache**.

```text
Prompt
  │
  ▼
┌──────────────────────┐
│ Transformer layers   │
└──────────┬───────────┘
           │
           ▼
       KV Cache
           │
           ▼
    Next-token generation
```

The KV cache avoids recomputing parts of the context for every generated token.

Its memory requirements increase with context length and concurrent requests.

## Project Structure

```text
Freellama/
├── Freellama/
│   ├── api/
│   │   ├── server.py
│   │   └── schemas.py
│   │
│   ├── engine/
│   │   ├── engine.py
│   │   ├── scheduler.py
│   │   └── worker.py
│   │
│   ├── models/
│   │   ├── loader.py
│   │   └── config.py
│   │
│   ├── backends/
│   │   ├── cpu.py
│   │   └── gpu.py
│   │
│   │
│   ├── generation/
│   │   ├── sampler.py
│   │   └── stopping.py
│   │
│   └── cli.py
│
├── tests/
├── examples/
├── benchmarks/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Run formatting:

```bash
ruff format .
```

Run static checks:

```bash
ruff check .
```

## Benchmarking

Run the included benchmark:

```bash
python benchmarks/generate.py \
  --model ./models/my-model \
  --tokens 512 \
  --requests 16
```

Example output:

```text
Requests:              16
Prompt tokens:         128
Generated tokens:      512
Total tokens:          10240

Time to first token:   91 ms
Throughput:            82.4 tok/s
Average latency:       6.2 s
Peak memory:           14.8 GB
```

Benchmark results are hardware- and configuration-dependent.

## API Compatibility

Freellama aims to support the commonly used OpenAI-compatible endpoints:

```text
GET  /v1/models

POST /v1/completions

POST /v1/chat/completions
```

This allows existing applications to point their model client at a local Freellama server with minimal changes.

For example:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="local"
)

response = client.chat.completions.create(
    model="my-model",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

## Security

By default, Freellama should bind to:

```text
127.0.0.1:8080
```

When exposing the server to a network, configure authentication, TLS, rate limiting, and appropriate network controls.

Do not expose an unauthenticated inference server directly to the public internet.

## Model Compatibility

Freellama is intended to support models using compatible transformer architectures.

Model compatibility depends on:

* Architecture
* Tensor formats
* Tokeniser
* Position encoding
* Attention implementation
* Quantization format
* Model-specific generation requirements

A model should not be assumed to work simply because its weights can be downloaded.

## Roadmap

* GPU backend
* CPU backend
* Continuous batching
* Paged KV cache
* Quantised inference
* Speculative decoding
* Multi-GPU inference
* OpenAI-compatible API
* Prometheus metrics
* Model hot-swapping
* Distributed inference

## Licence

See LICENCE for details.

## Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Add tests for your changes.
4. Run the test suite.
5. Submit a pull request.

Please keep performance-sensitive code benchmarked and documented.

---

**Freellama — run language models**
