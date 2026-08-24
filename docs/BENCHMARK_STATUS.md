# Benchmark status — 24 August 2026

This is Shashank Agnihotri's reconciled 32-model HSLE generation census after
applying the one-dispatch rule to previously observed blank or malformed
provider responses. A response is submitted once: an unusable first outcome is
an incorrect terminal result and is not retried.

## Executive census

- Models in scope: **32** across 13 families (23 multimodal and 9 text-only).
- Named settings: **4**; concrete response variants: **5**, because one-shot
  has fixed A and B assignments.
- Expected concrete prompt coordinates: **75,230**.
- Models complete across zero-shot, one-shot A, one-shot B, two-shot, and
  learning from experience: **24/32**.
- Models with a literal nonblank response at every coordinate: **23/32**.
- Real nonblank answers: **63,769**.
- Terminal incorrect settlements: **344**.
- Total scientifically settled coordinates: **64,113**.
- Scientifically safe-to-call coordinates: **11,044**.
- Paid/no-replay coordinates: **73**.
- Total unresolved coordinates: **11,117**.

Gemini 2.5 Pro explains the difference between the two completion counts: it
has 2,454 real answers and one authenticated terminal-incorrect LFE result.
DeepSeek-VL2 is strictly complete; its former 15 missing rows were recovered
as real answers.

## Completeness by concrete variant

Each variant has 15,046 expected coordinates: 23 multimodal models × 491
targets plus 9 text-only models × 417 text-only targets.

| Concrete variant | Complete models | Settled | Safely callable | Paid/no-replay | Unresolved |
|---|---:|---:|---:|---:|---:|
| Zero-shot | 26/32 | 13,006 | 2,025 | 15 | 2,040 |
| One-shot A | 26/32 | 13,019 | 2,011 | 16 | 2,027 |
| One-shot B | 26/32 | 13,005 | 2,021 | 20 | 2,041 |
| Two-shot | 26/32 | 13,004 | 2,026 | 16 | 2,042 |
| Learning from experience | 24/32 | 12,079 | 2,961 | 6 | 2,967 |
| **Total** | **24/32 complete across all five** | **64,113** | **11,044** | **73** | **11,117** |

## The 24 complete models

| Family | Complete model identifiers |
|---|---|
| Claude | `claude-haiku-4-5-20251001`, `claude-sonnet-4-6`, `claude-opus-4-7` |
| DeepSeek | `deepseek-ai/deepseek-vl2` |
| Gemini | `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.5-pro` |
| Gemma | `google/gemma-2-9b-it`, `google/gemma-3-27b-it`, `google/gemma-4-31B-it` |
| GPT | `gpt-5.1`, `gpt-5.2`, `gpt-5.4`, `gpt-5.5` |
| InternLM | `internlm/Intern-S1-mini` |
| Kimi | `moonshotai/Kimi-VL-A3B-Thinking` |
| Llama | `meta-llama/Meta-Llama-3-70B-Instruct`, `meta-llama/Llama-3.1-70B-Instruct`, `meta-llama/Llama-3.3-70B-Instruct` |
| LLaVA | `llava-hf/llava-onevision-qwen2-7b-ov-hf` |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.3` |
| Qwen | `Qwen2-VL-2B-Instruct`, `Qwen2.5-VL-32B-Instruct`, `Qwen3-VL-30B-A3B-Instruct` |

## Exact unanswered work by model

Each setting cell is **callable + paid/no-replay**. Paid/no-replay means a
request may already have incurred cost but lacks a trustworthy settlement, so
the portable runner will not replay it. Terminal results are already scored
incorrect and are not part of unanswered work.

| Incomplete model | Zero | One A | One B | Two | LFE | Callable | No-replay | Real answers | Terminal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `moonshotai/Kimi-K2-Thinking` | 295+1 | 290+2 | 305+2 | 296+2 | 251+0 | 1,437 | 7 | 640 | 1 |
| `moonshotai/Kimi-K2.5` | 435+0 | 435+2 | 445+1 | 440+2 | 490+1 | 2,245 | 6 | 169 | 35 |
| `moonshotai/Kimi-K2.6` | 480+0 | 485+0 | 484+2 | 482+1 | 490+1 | 2,421 | 4 | 23 | 7 |
| `moonshotai/Kimi-K3` | 355+11 | 370+8 | 365+10 | 383+8 | 490+1 | 1,963 | 38 | 401 | 53 |
| `qwen/qwen3.8-max` | 399+2 | 390+1 | 383+2 | 382+0 | 491+0 | 2,045 | 5 | 321 | 84 |
| `MiniMaxAI/MiniMax-M2.5` | 61+1 | 41+3 | 39+3 | 43+3 | 0+3 | 184 | 13 | 1,727 | 161 |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | 0+0 | 0+0 | 0+0 | 0+0 | 417+0 | 417 | 0 | 1,666 | 2 |
| `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5` | 0+0 | 0+0 | 0+0 | 0+0 | 332+0 | 332 | 0 | 1,753 | 0 |
| **Total** | **2,025+15** | **2,011+16** | **2,021+20** | **2,026+16** | **2,961+6** | **11,044** | **73** | **6,700** | **343** |

The 24 complete models contribute the other 57,070 settled coordinates:
57,069 real answers plus Gemini 2.5 Pro's one terminal result.

## Six-model OpenRouter workload

The third-cluster runner covers Kimi K2 Thinking, Kimi 2.5, Kimi 2.6, Kimi
K3, Qwen 3.8 Max, and MiniMax M2.5. It contains **10,295** callable
coordinates and binds **73** paid/no-replay coordinates.

| Route | Zero | One A | One B | Two | LFE | Coordinates | Generation turns |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kimi K2 Thinking | 295 | 290 | 305 | 296 | 251 | 1,437 | 1,939 |
| Kimi 2.5 | 435 | 435 | 445 | 440 | 490 | 2,245 | 3,225 |
| Kimi 2.6 | 480 | 485 | 484 | 482 | 490 | 2,421 | 3,401 |
| Kimi K3 | 355 | 370 | 365 | 383 | 490 | 1,963 | 2,943 |
| Qwen 3.8 Max | 399 | 390 | 383 | 382 | 491 | 2,045 | 3,027 |
| MiniMax M2.5 | 61 | 41 | 39 | 43 | 0 | 184 | 184 |
| **Total** | **2,025** | **2,011** | **2,021** | **2,026** | **2,212** | **10,295** | **14,719** |

The 2,212 LFE coordinates also require 4,424 Gemini 3.5 Flash feedback calls.
The runner therefore makes at most 19,143 single-dispatch API calls if every
callable coordinate reaches every planned turn. Final HLE correctness and
closeness judging is a separate project-owner step and is not included.

The authenticated v3 input bundle is in the complete Hugging Face dataset at
immutable revision `aeda08b2536a19e698d027fd4f701eea78c9171d`. Its
SHA-256 is
`ced06f31b7d82a58db28391f6e9bf09293a88933480f6b5354784ce98d3ede5f`.

See [the one-command OpenRouter workflow](OPENROUTER_RESUME.md) and
[the corrected cost audit](OPENROUTER_COSTS.md).

## What is not complete

Eight models still have unresolved generation coordinates: the six
OpenRouter models above and the two Nemotron models. Generation completion is
also not final paper completion. A complete 32-model matrix with both HLE
correctness and closeness judgments has not yet been assembled, and final
plots have not yet been produced. The metric code rejects missing coordinates
instead of fabricating scores.
