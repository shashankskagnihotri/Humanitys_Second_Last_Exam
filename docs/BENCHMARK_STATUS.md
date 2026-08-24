# Benchmark status — 24 August 2026

This is the reconciled generation census for the intended 33-model HSLE
scope. It distinguishes real answers, terminal policy settlements, safely
callable work, and paid requests that must not be replayed. Those categories
must not be collapsed into a single “answered” count.

## Executive census

- Models in scope: **33** across 13 families (23 multimodal and 10 text-only).
- Expected concrete prompt coordinates: **77,315**.
- Models complete across zero-shot, both one-shot assignments, two-shot, and
  learning from experience: **24/33**.
- Models with a literal nonblank response at every coordinate: **23/33**.
- Real nonblank answers: **63,769**.
- Authenticated terminal-policy settlements: **325**.
- Total scientifically settled coordinates: **64,094**.
- Scientifically safe-to-call coordinates: **13,148**, conditional on route,
  credit, cluster, and owner authorization.
- Paid/no-replay coordinates: **73**.
- Total unresolved coordinates: **13,221**.

Gemini 2.5 Pro explains the difference between the two complete-model counts:
it has 2,454 nonblank answers and one authenticated terminal-WRONG LFE
coordinate. DeepSeek-VL2 is now strictly complete; its former 15 missing rows
were recovered as real nonblank answers, so no DeepSeek imputation remains.

## Completeness by concrete variant

Each variant has 15,463 expected coordinates: 23 multimodal models × 491
targets plus 10 text-only models × 417 targets.

| Concrete variant | Complete models | Settled | Safely callable | Paid/no-replay | Unresolved |
|---|---:|---:|---:|---:|---:|
| Zero-shot | 26/33 | 12,998 | 2,450 | 15 | 2,465 |
| One-shot A | 26/33 | 13,017 | 2,430 | 16 | 2,446 |
| One-shot B | 26/33 | 12,999 | 2,444 | 20 | 2,464 |
| Two-shot | 26/33 | 13,002 | 2,445 | 16 | 2,461 |
| Learning from experience | 24/33 | 12,078 | 3,379 | 6 | 3,385 |
| **Total** | **24/33 complete across all five** | **64,094** | **13,148** | **73** | **13,221** |

The four named experimental settings correspond to five concrete response
variants because one-shot has two prespecified assignments, A and B.

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

## Exact work remaining by model

Each setting cell is **safely callable + paid/no-replay**. A no-replay item is
unresolved, but the prior paid dispatch is ambiguous and therefore must not be
submitted automatically. “Settled” includes real nonblank answers and
authenticated terminal outcomes.

| Incomplete model | Zero | One A | One B | Two | LFE | Callable | No-replay | Real answers | Terminal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `moonshotai/Kimi-K2-Instruct` | 417+0 | 417+0 | 417+0 | 417+0 | 417+0 | 2,085 | 0 | 0 | 0 |
| `moonshotai/Kimi-K2-Thinking` | 295+1 | 290+2 | 305+2 | 296+2 | 252+0 | 1,438 | 7 | 640 | 0 |
| `moonshotai/Kimi-K2.5` | 436+0 | 435+2 | 445+1 | 440+2 | 490+1 | 2,246 | 6 | 169 | 34 |
| `moonshotai/Kimi-K2.6` | 480+0 | 485+0 | 484+2 | 482+1 | 490+1 | 2,421 | 4 | 23 | 7 |
| `moonshotai/Kimi-K3` | 359+11 | 372+8 | 370+10 | 384+8 | 490+1 | 1,975 | 38 | 401 | 41 |
| `qwen/qwen3.8-max` | 400+2 | 390+1 | 383+2 | 383+0 | 491+0 | 2,047 | 5 | 321 | 82 |
| `MiniMaxAI/MiniMax-M2.5` | 63+1 | 41+3 | 40+3 | 43+3 | 0+3 | 187 | 13 | 1,727 | 158 |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | 0+0 | 0+0 | 0+0 | 0+0 | 417+0 | 417 | 0 | 1,666 | 2 |
| `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5` | 0+0 | 0+0 | 0+0 | 0+0 | 332+0 | 332 | 0 | 1,753 | 0 |
| **Total** | **2,450+15** | **2,430+16** | **2,444+20** | **2,445+16** | **3,379+6** | **13,148** | **73** | **6,700** | **324** |

“Callable” is a scientific replay classification, not blanket launch
authorization. Kimi K2 Instruct's 2,085 coordinates are outside the public
runner. MiniMax's 187 are cap-held and require added credit or new owner
authority. The 417 Nemotron Nano and 332 Nemotron Super coordinates retain
their local-cluster routes. The five-route OpenRouter runner covers the
remaining 10,127 coordinates described below.

The 24 complete models contribute the other 57,070 settled coordinates:
57,069 real nonblank answers and Gemini 2.5 Pro's one terminal-WRONG row.

## Public five-route OpenRouter handoff

The portable cluster runner covers Kimi K2 Thinking, Kimi 2.5, Kimi 2.6,
Kimi K3, and Qwen 3.8 Max. It contains **10,127** safely callable coordinates
and excludes **60** paid/no-replay coordinates. The callable split is:

| Route | Zero | One A | One B | Two | LFE | Total |
|---|---:|---:|---:|---:|---:|---:|
| Kimi K2 Thinking | 295 | 290 | 305 | 296 | 252 | 1,438 |
| Kimi 2.5 | 436 | 435 | 445 | 440 | 490 | 2,246 |
| Kimi 2.6 | 480 | 485 | 484 | 482 | 490 | 2,421 |
| Kimi K3 | 359 | 372 | 370 | 384 | 490 | 1,975 |
| Qwen 3.8 Max | 400 | 390 | 383 | 383 | 491 | 2,047 |
| **Total** | **1,970** | **1,972** | **1,987** | **1,985** | **2,213** | **10,127** |

The workload represents 14,553 logical model calls: 7,914 static calls plus
three generation calls for each of 2,213 LFE coordinates. LFE also requires
4,426 separately relayed Gemini feedback decisions. The public runner never
replays a settled or paid-ambiguous request and never performs final HLE or
closeness judging.

The authenticated input archive is published in the public Hugging Face
dataset at immutable revision
`6861ef237eb9501b8fda3d4fe61788154e143c22`. The archive SHA-256 is
`226ba161608182f37ee4310bd8d3cb32457604603f272ad3a88ee5d0666ecd23`.

## What is not complete

Generation completion is not the same as final paper completion. There is no
authoritative assembled 33-model matrix containing both HLE correctness and
closeness judgments, and the final plot set has not been produced. Current
metric code therefore requires a complete judged matrix and does not fabricate
scores for missing generation or judge rows.

At the audit snapshot there were no active matching scheduler jobs. Kimi K3's
exact Morph route was degraded during live preflight, so a third-cluster launch
must remain blocked until the runner's exact-route health check passes. The
other four public routes were visible as healthy at that snapshot. No provider
generation call or real Slurm job was submitted while preparing this release.
