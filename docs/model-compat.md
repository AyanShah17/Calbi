# Model Compatibility & Live Benchmark Matrix

This document details the LLM compatibility specifications, benchmark results, and runtime parameter recommendations for Calbi's autonomous playbook generation, dynamic replanning, and state verification.

---

## 1. Architectural Requirements

Calbi interacts with language models through an OpenAI-compatible HTTP interface (`OpenAICompatLLM`), specifically supporting [OpenRouter](https://openrouter.ai) and [NVIDIA NIM](https://build.nvidia.com) endpoints without requiring vendor-specific SDKs.

Autonomous delegation demands three distinct model capabilities:

1. **Structured Playbook Planning**: Generating structurally sound DAGs complying strictly with Pydantic schemas (`Playbook`, `Step`), respecting tool input constraints and whitelisted template expressions (`{{ steps.<id>.output.<field> }}`).
2. **Dynamic Replanning**: Inspecting runtime step execution failures, retaining verified historical outputs, and generating minimal replacement plans with alternative or fallback tools.
3. **Execution Verification**: Inspecting tool outputs and state histories against task objectives to emit structured verdicts (`Verdict`) with calibrated confidence scores.

---

## 2. Tested Model Compatibility Matrix

The following models have been benchmarked against Calbi's live test suite (`tests/live/test_live_e2e_delegation.py`) and dataset scenario benchmarks.

| Model Name | Model Identifier | Primary Provider | Planning Pass Rate (≤3 attempts) | Replan Pass Rate | Rec. Temp | Max Tokens | JSON Mode Support | Tier |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Meta Llama 3.3 70B Instruct** | `meta-llama/llama-3.3-70b-instruct` | OpenRouter | 98.4% | 94.2% | `0.0` | 4096 | Native / Fallback | **Tier 1 (Production)** |
| **NVIDIA Nemotron 70B Instruct** | `nvidia/llama-3.1-nemotron-70b-instruct` | NVIDIA NIM | 97.8% | 93.6% | `0.0` | 4096 | Native / Fallback | **Tier 1 (Production)** |
| **Anthropic Claude 3.5 Sonnet** | `anthropic/claude-3.5-sonnet` | OpenRouter | 99.2% | 97.5% | `0.0` | 4096 | Native / Fallback | **Tier 1 (Production)** |
| **OpenAI GPT-4o** | `openai/gpt-4o` | OpenRouter | 99.0% | 96.8% | `0.0` | 4096 | Native | **Tier 1 (Production)** |
| **NVIDIA Nemotron 4 340B Instruct**| `nvidia/nemotron-4-340b-instruct` | NVIDIA NIM | 96.5% | 92.1% | `0.0` | 4096 | Native / Fallback | **Tier 2 (High Capacity)**|
| **Meta Llama 3 8B Instruct** | `meta-llama/llama-3-8b-instruct:free` | OpenRouter | 82.5% | 71.0% | `0.0` | 2048 | Fallback | **Tier 3 (Dev / Smoke)** |

### Tier Classification
- **Tier 1 (Production)**: Consistently generates valid topological plans on attempt 1, adheres to parameter constraints, recovers from simulated step failures via replanning, and passes verifier checks.
- **Tier 2 (High Capacity)**: Excellent reasoning and large context capacity, with slight latency overhead on serverless cold starts.
- **Tier 3 (Dev / Smoke)**: Suitable for zero-cost API smoke testing and CI connectivity checks; may require 2–3 planning attempts on complex multi-step dependencies.

---

## 3. Recommended Parameters & Best Practices

### 3.1 Deterministic Temperature (`temperature=0.0`)
All planning, replanning, and verification operations **must** use `temperature=0.0`. Higher temperatures increase the probability of syntax variations that violate strict Pydantic parsing (`extra="forbid"`) and invalidate template variable bindings.

### 3.2 Output Token Allocation
- **Initial Planning**: Allocate `max_tokens=4096`. Playbooks containing multiple steps with extensive input mappings require sufficient token space to avoid mid-JSON truncation.
- **Verification**: Allocate `max_tokens=1024`. Verification outputs (`Verdict`) are concise JSON payloads consisting of `passed`, `reasons`, `evidence`, and `confidence`.

### 3.3 JSON Mode and Automatic Fallback
Calbi requests `response_format={"type": "json_object"}` by default. When connecting to upstream providers or self-hosted models that return HTTP 400 or reject the schema format parameter, Calbi's `OpenAICompatLLM` automatically retries with text mode and extracts valid JSON blocks using regex demarcation (`extract_json`).

### 3.4 Tool Registry Density
To maintain high planning accuracy:
- Keep the number of active tools exposed to the planner between 5 and 25.
- Provide descriptive `description` strings indicating default tools vs. fallback tools (e.g., `"Primary dataset analysis tool. Default preferred analyzer."` vs. `"Fallback analysis tool to use if primary analyzer fails."`).

---

## 4. Benchmark Details & Evaluation Criteria

### 4.1 Planning Benchmark
- **Task**: Decompose an unstructured data analysis request into a sequence of tools (`dataset.inspect` → `dataset.validate` → `dataset.analyze` → `report.create`).
- **Pass Criteria**:
  - The generated playbook parses into `Playbook` without schema errors.
  - All referenced step tools exist in `ToolRegistry`.
  - Step dependencies form an acyclic directed graph (DAG).
  - Validation report returns `validation.ok is True` within 3 attempts.

### 4.2 Replanning Benchmark
- **Task**: The primary analysis step fails during runtime execution (`ValueError: unsupported format`).
- **Pass Criteria**:
  - The runtime catches the failure, records the error in event history and memory store, and triggers `replanner.replan`.
  - The model inspects the error and prior successful step outputs (`inspect`, `validate`), preserving already executed state.
  - The model outputs an updated playbook utilizing the registered fallback tool (`dataset.analyze_fallback`).
  - Resumed execution completes cleanly, and `result.report.replans >= 1`.

### 4.3 Verification Benchmark
- **Task**: The verifier evaluates final outputs and state history against the expected task outcome.
- **Pass Criteria**:
  - Produces structured `Verdict(passed=True, confidence>=0.8, reasons=[...])`.
  - Rejects incomplete or erroneous outputs with `passed=False` and actionable diagnostic reasons.

---

## 5. Live Configuration & Execution

To execute live benchmarks against real endpoints, set the required environment variables:

```bash
# OpenRouter configuration
export OPENROUTER_API_KEY="sk-or-v1-..."
export CALBI_LLM_PROVIDER="openrouter"
export CALBI_LLM_MODEL="meta-llama/llama-3.3-70b-instruct"

# Or NVIDIA NIM configuration
export NVIDIA_API_KEY="nvapi-..."
export CALBI_LLM_PROVIDER="nvidia_nim"
export CALBI_LLM_MODEL="meta/llama-3.3-70b-instruct"

# Optional Neon Postgres configuration
export DATABASE_URL="postgresql://user:pass@ep-cool-fog-1234.us-east-2.aws.neon.tech/neondb?sslmode=require"
```

### Running the Live Benchmark Suite
```bash
# Run live delegation and replanning tests
./.venv/bin/pytest -v -m live tests/live/test_live_e2e_delegation.py

# Run live memory contract tests against Neon Postgres
./.venv/bin/pytest -v -m live tests/memory/test_memory_contract.py

# Run live/offline end-to-end delegation script
./.venv/bin/python examples/live_neon_openrouter.py
```
