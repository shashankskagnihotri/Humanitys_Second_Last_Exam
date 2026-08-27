# Benchmark status — 27 August 2026

This is Shashank Agnihotri's reconciled 33-model HSLE generation census after
applying the one-dispatch rule to previously observed blank or malformed
provider responses. A response is submitted once: an unusable first outcome is
an incorrect terminal result and is not retried.

## Executive census

- Models in scope: **33** across 13 families (24 multimodal and 9 text-only).
- Named settings: **4**; concrete response variants: **5**, because one-shot
  has fixed A and B assignments.
- Expected concrete prompt coordinates: **77,685**.
- Models complete across zero-shot, one-shot A, one-shot B, two-shot, and
  learning from experience: **24/33**.
- Models with a literal nonblank response at every coordinate: **23/33**.
- Real nonblank answers: **63,443**.
- Terminal incorrect settlements: **449**.
- Total scientifically settled coordinates: **63,892**.
- Scientifically safe-to-call coordinates: **13,725**.
- Paid/no-replay coordinates: **68**.
- Total unresolved coordinates: **13,793**.

Gemini 2.5 Pro explains the difference between the two completion counts: it
has 2,454 real answers and one authenticated terminal-incorrect LFE result.
DeepSeek-VL2 is strictly complete; its former 15 missing rows were recovered
as real answers.

## Completeness by concrete variant

Each variant has 15,537 expected coordinates: 24 multimodal models × 491
targets plus 9 text-only models × 417 text-only targets.

| Concrete variant | Complete models | Settled | Safely callable | Paid/no-replay | Unresolved |
|---|---:|---:|---:|---:|---:|
| Zero-shot | 26/33 | 12,977 | 2,547 | 13 | 2,560 |
| One-shot A | 26/33 | 12,960 | 2,562 | 15 | 2,577 |
| One-shot B | 26/33 | 12,938 | 2,581 | 18 | 2,599 |
| Two-shot | 26/33 | 12,938 | 2,583 | 16 | 2,599 |
| Learning from experience | 24/33 | 12,079 | 3,452 | 6 | 3,458 |
| **Total** | **24/33 complete across all five** | **63,892** | **13,725** | **68** | **13,793** |

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
| `Qwen/Qwen3.8-27B` | 491+0 | 491+0 | 491+0 | 491+0 | 491+0 | 2,455 | 0 | 0 | 0 |
| `MiniMaxAI/MiniMax-M2.5` | 0+1 | 0+3 | 0+3 | 0+3 | 0+3 | 0 | 13 | 1,722 | 350 |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | 0+0 | 0+0 | 0+0 | 0+0 | 417+0 | 417 | 0 | 1,666 | 2 |
| `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5` | 0+0 | 0+0 | 0+0 | 0+0 | 332+0 | 332 | 0 | 1,753 | 0 |
| `Qwen/Qwen3.8-Flash-Next` | 491+0 | 491+0 | 491+0 | 491+0 | 491+0 | 2,455 | 0 | 0 | 0 |
| **Total** | **2,547+13** | **2,562+15** | **2,581+18** | **2,583+16** | **3,452+6** | **13,725** | **68** | **6,374** | **448** |

The Qwen3.8 Flash Next row is the exact multimodal
`Qwen/Qwen3.8-Flash-Next` checkpoint at immutable revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`, with official default
`reasoning_effort=xhigh`. It is newly in scope, has zero pre-existing
responses, and is not classified complete.

The 24 complete models contribute the other 57,070 settled coordinates:
57,069 real answers plus Gemini 2.5 Pro's one terminal result.

MiniMax has no safely callable work left. Its 13 no-replay coordinates are
operational evidence gaps, not model responses, and are not safe to resubmit.
The zero-retry reconciliation reclassifies 149 responses obtained only on a
later historical attempt as terminal incorrect; this is why MiniMax now has
1,722 valid first responses and 350 terminal incorrect outcomes.

## Active Helix local workload

The collaborator workflow creates a new full local evidence layer from exact
pinned checkpoints; it does not mix or replay the historical hosted-provider
rows in the census above. Four models fit the documented one-node Helix
contract and are launchable now.

| Route | Static coordinates | LFE coordinates | Maximum model calls | Maximum feedback calls | Helix state |
|---|---:|---:|---:|---:|---|
| Kimi K2 Thinking | 1,668 | 417 | 2,919 | 834 | Launchable, 8×H200 TP8 |
| Kimi K2.5 | 1,964 | 491 | 3,437 | 982 | Launchable, 8×H200 TP8 |
| Kimi K2.6 | 1,964 | 491 | 3,437 | 982 | Launchable, 8×H200 TP8 |
| Qwen3.8 27B | 1,964 | 491 | 3,437 | 982 | Launchable, 8×H200 TP8 |
| **Launchable total** | **7,560** | **1,890** | **13,230** | **3,780** | **9,450 coordinates** |
| Kimi K3 | 1,964 | 491 | 3,437 | 982 | Blocked: official ~1,680-GB GPU footprint exceeds one 8×H200 node |

Every static coordinate has one model call. Each LFE coordinate has at most
three ordered model calls and two Gemini 3.5 Flash feedback calls. Every call
has immutable write-ahead state and at most one request attempt. Final HLE
correctness and closeness judging is a separate project-owner step.

Kimi K3 will not be submitted through the one-node path: the verified vLLM
recipe is multi-node with the official CUDA 13 container/nightly runtime, and
an unverified CPU-offload substitute would not be scientifically defensible.
The Qwen route is the native-multimodal `Qwen/Qwen3.8-27B` checkpoint, not the
former hosted Max identity. See [the Helix contract](HELIX_LOCAL.md).

## Current 25-model scored plot release

The current scored release contains **25 models across 12 represented
families**: all 24 generation-complete models plus MiniMax M2.5 under an
explicit conservative lower-bound projection. Inclusion in this release does
not reclassify MiniMax's 13 paid/no-replay coordinates as real responses, and
it does not claim that any unfinished Kimi, Qwen, or Nemotron model is complete.

The release validates **59,155 canonical concrete coordinates**. Generation
evidence is present for 59,142; HLE score evidence is present for 59,141; and
effective closeness score evidence is present for 59,142. MiniMax's sole
otherwise-missing HLE verdict was closed by Gemini 3.5 Flash with an explicit
formula-preserving constrained fallback; the exact model answer was not
changed. Missing metric evidence is conservatively scored zero only in the
plotted projection. Both Kimi-VL-A3B-Thinking and MiniMax M2.5 remain labeled
“lower bound” in both metrics because their generation layers contain
disclosed terminal or no-replay gaps; this does not imply that all of their
underlying score evidence is missing.

The all-model summaries contain 47,324 logical model-question-setting units
and use each model's native 491-question multimodal or 417-question text-only
cohort. Family figures follow the comparability rule requested for the paper:
if any model in a family is text-only, every model in that family is compared
on the same 417 text questions; otherwise all family members use the same 491
questions. That family projection contains 58,415 concrete coordinates and
46,732 logical units. One-shot A and B are paired within the original question
before aggregation.

The final inventory is exactly **26 one-page PDFs**: two all-model plots and
24 family plots. It contains no PNG or SVG files and no titles, subtitles,
footnotes, audit prose, or explanatory annotations. The public plotting code
reproduces this exact cohort, aggregation, styling, and PDF-only contract from
the coordinate-level scored layer; generated score tables and figures remain
outside Git.

## What is not complete

Nine models still have unresolved generation coordinates: four Kimi models,
Qwen3.8 27B, Qwen3.8 Flash Next, MiniMax's 13 no-replay gaps, and the two
Nemotron models. Generation completion is also not final paper completion. A
complete 33-model matrix with both HLE correctness and closeness judgments has not yet
been assembled. The 25-model PDF release described above is current, but it
does not include or imply completion of any unfinished Kimi, Qwen, or Nemotron
model. The general metric code rejects missing coordinates
instead of fabricating scores; the disclosed lower-bound plot projection is a
separate, explicit publication policy.
