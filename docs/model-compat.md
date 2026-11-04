# Model Compatibility Guide

Calbi works with any OpenAI-compatible API that supports chat completions and JSON output.

## Tested Providers

### 1. OpenRouter (`OpenRouterLLM`)
- Base URL: `https://openrouter.ai/api/v1`
- Recommended Models:
  - `meta-llama/llama-3.3-70b-instruct` (Fast, high schema compliance)
  - `anthropic/claude-3.5-sonnet` (Best for complex replanning)
  - `openai/gpt-4o` (High reasoning accuracy)

### 2. NVIDIA NIM (`NvidiaNimLLM`)
- Base URL: `https://integrate.api.nvidia.com/v1` (or local self-hosted NIM via `NVIDIA_NIM_BASE_URL`)
- Recommended Models:
  - `meta/llama-3.3-70b-instruct`
  - `nvidia/nemotron-4-340b-instruct`
