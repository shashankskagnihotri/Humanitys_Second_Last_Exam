# Benchmark status — 25 August 2026

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
- Real nonblank answers: **63,764**.
- Terminal incorrect settlements: **533**.
- Total scientifically settled coordinates: **64,297**.
- Scientifically safe-to-call coordinates: **10,860**.
- Paid/no-replay coordinates: **73**.
- Total unresolved coordinates: **10,933**.

Gemini 2.5 Pro explains the difference between the two completion counts: it
has 2,454 real answers and one authenticated terminal-incorrect LFE result.
DeepSeek-VL2 is strictly complete; its former 15 missing rows were recovered
as real answers.

## Completeness by concrete variant

Each variant has 15,046 expected coordinates: 23 multimodal models × 491
targets plus 9 text-only models × 417 text-only targets.

| Concrete variant | Complete models | Settled | Safely callable | Paid/no-replay | Unresolved |
|---|---:|---:|---:|---:|---:|
| Zero-shot | 26/32 | 13,067 | 1,964 | 15 | 1,979 |
| One-shot A | 26/32 | 13,060 | 1,970 | 16 | 1,986 |
| One-shot B | 26/32 | 13,044 | 1,982 | 20 | 2,002 |
| Two-shot | 26/32 | 13,047 | 1,983 | 16 | 1,999 |
| Learning from experience | 24/32 | 12,079 | 2,961 | 6 | 2,967 |
| **Total** | **24/32 complete across all five** | **64,297** | **10,860** | **73** | **10,933** |

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
| `MiniMaxAI/MiniMax-M2.5` | 0+1 | 0+3 | 0+3 | 0+3 | 0+3 | 0 | 13 | 1,722 | 350 |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | 0+0 | 0+0 | 0+0 | 0+0 | 417+0 | 417 | 0 | 1,666 | 2 |
| `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5` | 0+0 | 0+0 | 0+0 | 0+0 | 332+0 | 332 | 0 | 1,753 | 0 |
| **Total** | **1,964+15** | **1,970+16** | **1,982+20** | **1,983+16** | **2,961+6** | **10,860** | **73** | **6,695** | **532** |

The 24 complete models contribute the other 57,070 settled coordinates:
57,069 real answers plus Gemini 2.5 Pro's one terminal result.

MiniMax has no safely callable work left. Its 13 no-replay coordinates are
operational evidence gaps, not model responses, and are not safe to resubmit.
The zero-retry reconciliation reclassifies 149 responses obtained only on a
later historical attempt as terminal incorrect; this is why MiniMax now has
1,722 valid first responses and 350 terminal incorrect outcomes.

## Five-model OpenRouter workload

The third-cluster runner covers Kimi K2 Thinking, Kimi 2.5, Kimi 2.6, Kimi
K3, and Qwen 3.8 Max. It contains **10,111** callable coordinates and binds
**60** paid/no-replay coordinates. MiniMax is excluded from launcher and worker
route choices.

| Route | Zero | One A | One B | Two | LFE | Coordinates | Generation turns |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kimi K2 Thinking | 295 | 290 | 305 | 296 | 251 | 1,437 | 1,939 |
| Kimi 2.5 | 435 | 435 | 445 | 440 | 490 | 2,245 | 3,225 |
| Kimi 2.6 | 480 | 485 | 484 | 482 | 490 | 2,421 | 3,401 |
| Kimi K3 | 355 | 370 | 365 | 383 | 490 | 1,963 | 2,943 |
| Qwen 3.8 Max | 399 | 390 | 383 | 382 | 491 | 2,045 | 3,027 |
| **Total** | **1,964** | **1,970** | **1,982** | **1,983** | **2,212** | **10,111** | **14,535** |

The 2,212 LFE coordinates also require 4,424 Gemini 3.5 Flash feedback calls.
The runner therefore makes at most 18,959 single-dispatch API calls if every
callable coordinate reaches every planned turn. Final HLE correctness and
closeness judging is a separate project-owner step and is not included.

The authenticated v3 input bundle is in the complete Hugging Face dataset at
immutable revision `aeda08b2536a19e698d027fd4f701eea78c9171d`. Its
SHA-256 is
`ced06f31b7d82a58db28391f6e9bf09293a88933480f6b5354784ce98d3ede5f`.
Those archive bytes remain unchanged after the archived Sail Research K3
endpoint disappeared. The live runner records and authenticates a Git-bound
successor to DeepInfra BF16 without changing any question, prompt, image,
exclusion, or task vector. The archive retains MiniMax provenance, but the live
runner does not expose a MiniMax selection.

See [the one-command OpenRouter workflow](OPENROUTER_RESUME.md) and
[the corrected cost audit](OPENROUTER_COSTS.md).

## What is not complete

Eight models still have unresolved generation coordinates: the five callable
OpenRouter models above, MiniMax's 13 no-replay gaps, and the two Nemotron
models. Generation completion is
also not final paper completion. A complete 32-model matrix with both HLE
correctness and closeness judgments has not yet been assembled, and final
plots have not yet been produced. The metric code rejects missing coordinates
instead of fabricating scores.
