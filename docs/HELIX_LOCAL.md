# Helix local H200 benchmark handoff

This is the active collaborator workflow for capacity-compatible, locally
released Kimi checkpoints. It performs no computation on a login node and uses
no hosted model-routing service. A launchable entry command resolves
workspace-backed storage, submits a CPU preparation job, and submits one
dependent H200 generation job.

## Prerequisites

- Helix environment modules and Slurm commands are available on the login node.
- The official 10-TB workspace service is available through `ws_find` and
  `ws_allocate`, or `HSLE_HELIX_WORKSPACE` names an existing workspace.
- `HSLE_GEMINI_KEY_FILE` names a regular, non-symlink, user-owned file with no
  group/other permissions and is visible from compute nodes. It contains
  exactly one Gemini API key for inline Gemini 3.5 Flash binary LFE feedback.
  The key is never placed on a command line or written to a result.
- The pinned model and dataset snapshots are public. Preparation downloads
  without authentication, and preparation does not inherit a Hugging Face
  credential.

Example key-file preparation:

```bash
install -m 600 /dev/null /secure/path/gemini.key
# Put the key on the single line of that file using your secret-management tool.
export HSLE_GEMINI_KEY_FILE=/secure/path/gemini.key
```

Optional site-specific variables are:

| Variable | Meaning | Default |
|---|---|---|
| `HSLE_HELIX_WORKSPACE` | Existing workspace path | Resolve/allocate by name |
| `HSLE_HELIX_WORKSPACE_NAME` | Workspace-service name | `hsle-benchmark-$USER` |
| `HSLE_HELIX_ACCOUNT` | Slurm account | Slurm site default |
| `HSLE_HELIX_CPU_PARTITION` | Preparation partition | `cpu-single` |
| `HSLE_HELIX_PYTHON_MODULE` | Python module | `devel/python` |
| `HSLE_HELIX_CUDA_MODULE` | CUDA module | `devel/cuda` |
| `HSLE_HELIX_CONCURRENCY` | Independent in-flight coordinates | `16` |

## One-command entries

Run one command for each desired exact checkpoint:

```bash
bash script/run_helix_kimi_k26.sh
bash script/run_helix_kimi_k25.sh
bash script/run_helix_kimi_k2_thinking.sh
bash script/run_helix_qwen38_27b.sh
```

Each command prints the CPU job ID, its dependent GPU job ID, the workspace,
and the result directory. CPU preparation requests one node, 16 CPUs, 128 GB,
and at most 120 hours. Generation requests exactly one exclusive `gpu-single`
node, eight H200 GPUs, 64 CPUs, 2,100 GB, and at most 120 hours. Neither job is
requeued. The GPU job starts only after successful preparation.

Both submissions use `--export=NONE`. Module names, concurrency, and the
non-secret protected-key file path are passed explicitly to the workers, so
site overrides remain deterministic without exporting the login environment.

Preparation creates a route-specific virtual environment, installs the pinned
vLLM runtime, downloads the exact public model and dataset revisions without
authentication, then validates the model and data before generation. The
public dataset authority is:

- repository `shashankskagnihotri/humanitys-second-last-exam`;
- revision `aeda08b2536a19e698d027fd4f701eea78c9171d`;
- consolidated-table SHA-256
  `8a0568576ed1788e21899171c4e5a379b814ac09d912a26e3ad8a835a0337b04`;
- 491 targets, 982 context instances, 491 two-example linkage rows, and 258
  canonical files below `data/images/`.

Legacy `data/image/` references are normalized to `data/images/`, and aliases
are deduplicated before the exact image universe is checked.
See [correction provenance](CORRECTIONS.md) for the pinned curation layer.

## Exact local identities

| Entry | Hugging Face model | Revision | Scope |
|---|---|---|---:|
| Kimi K2 Thinking | `moonshotai/Kimi-K2-Thinking` | `a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55` | 2,085 text-only coordinates |
| Kimi K2.5 | `moonshotai/Kimi-K2.5` | `4d01dfe0332d63057c186e0b262165819efb6611` | 2,455 multimodal coordinates |
| Kimi K2.6 | `moonshotai/Kimi-K2.6` | `7eb5002f6aadc958aed6a9177b7ed26bb94011bb` | 2,455 multimodal coordinates |
| Kimi K3 | `moonshotai/Kimi-K3` | `9f62e4e9fffbd0a83ddd60e1c209d828994b3569` | 2,455 multimodal coordinates |
| Qwen3.8 27B | `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 2,455 multimodal coordinates |

The K3 snapshot contains exactly 1,560,936,091,448 bytes of safetensors
checkpoint weights. Its official vLLM footprint is about 1,680 GB of GPU
memory, which exceeds the roughly 1,128 GB available on a single eight-H200
Helix node. The verified recipe is multi-node and requires the official CUDA
13 container/nightly runtime and driver R580 or newer. Consequently:

```bash
bash script/run_helix_kimi_k3.sh
```

fails before workspace creation or job submission. It will remain guarded
until Helix provides a documented 16-GPU multi-node allocation and the exact
official runtime recipe is implemented. The source runner already encodes K3's
chat identity requirement (`reasoning_effort=max` and preservation of complete
assistant `reasoning_content` in LFE history), but that is not a claim that the
current one-node runtime can serve the checkpoint. Preserve-thinking behavior
is not enabled for K2.6.

The four launchable models are served on tensor parallelism 8 with maximum
output 4,096, maximum model length 32,768, seed 0, temperature 0, and top-p
1.0. The runtime records the exact model revision, vLLM version, node, job,
generation settings, and token usage. The vLLM identity preflight and every
successful response must report the exact requested model ID.
Before vLLM starts, the GPU job recomputes the SHA-256 of every safetensors
shard plus the model and tokenizer configuration files and compares them with
the signed preparation authority. Generation also requires that verification
to belong to the current Slurm job, so a mutable or corrupted local snapshot
cannot be reported as the pinned checkpoint.

## Qwen3.8 27B local identity

The Qwen entry is also present:

```bash
bash script/run_helix_qwen38_27b.sh
```

This is a runnable one-node TP8 route for the exact native-multimodal
`Qwen/Qwen3.8-27B` checkpoint at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`: 18 safetensors totaling
55,563,006,776 bytes, `Qwen3_5ForConditionalGeneration`, Apache-2.0. It uses
vLLM 0.25.1 with the `qwen3` reasoning parser, multimodal encoder data
parallelism, and the exact three-image HSLE envelope. Thinking remains enabled
with `reasoning_effort=xhigh` and prior reasoning is preserved in LFE history.
The study keeps temperature 0, top-p 1, seed 0, and 4,096 output tokens for
cross-model comparability; this intentionally differs from optional sampling
recommendations on the model card. This identity does not inherit any response
or terminal evidence from the former hosted Max route.

## One-attempt and resume contract

Every static coordinate has exactly one target-generation call. Each LFE
coordinate preserves its required order internally: example 1, optional valid
response feedback, example 2, optional valid response feedback, then target.
Each model call and each eligible feedback call has an immutable write-ahead
intent followed by an immutable result. There are no request retry loops, SDK
retries, fallback providers, or model substitutions.

Inline feedback uses the public `render_hle_judge_prompt` template and one
Gemini 3.5 Flash request. The request asks for the exact structured HLE fields
`extracted_final_answer`, `reasoning`, `correct`, `confidence`, and `strict`;
requires `correct` to be `yes` or `no`, confidence to be an integer from 0
through 100, and `strict` to be true; and rejects extra fields. Dispatch uses
JSON response mode, seed 0, one candidate, 8,192 maximum output tokens, LOW
thinking, no sampling knobs, and `BLOCK_NONE` safety thresholds. The study's
feedback identity label remains medium, while the recorded wire contract
preserves the actual LOW dispatch configuration. Feedback is accepted only
when raw `modelVersion` is exactly `gemini-3.5-flash`. The complete request
payload, response, and contract are recorded without the credential.

A blank, malformed, rejected, timed-out, or failed first model outcome is
terminal incorrect. If interruption leaves an intent without a result, the
attempt becomes ambiguous/no-replay; restarting does not issue it again.
Coordinates without an intent remain resumable. Independent coordinates run
with bounded concurrency, but a failure in one future does not cancel or
replay any other future. CSV export is serialized by the main thread and uses
atomic replacement.

The generation command exits nonzero if a coordinate is operationally
incomplete or if an uncalled coordinate remains. Authenticated terminal
first-model-response failures are valid incorrect settlements and do not by
themselves make the job fail.

Results are stored under:

```text
<workspace>/hsle-local-benchmark/need_to_be_judged/<route>/
├── wal/<evaluation-key>/
├── generation_results/<evaluation-key>.json
├── responses.csv
└── generation_summary.json
```

The per-coordinate records include full prompts, question/example IDs, image
digests, content, reasoning content, raw response, served/requested identity,
revision, runtime settings, usage, and terminal status. Final HLE accuracy and
closeness judging are intentionally deferred and are not run on Helix by this
workflow.

## Authoritative platform and model references

- [Helix hardware](https://wiki.bwhpc.de/e/Helix/Hardware)
- [Helix Slurm contract](https://wiki.bwhpc.de/e/Helix/Slurm)
- [Helix filesystems and workspaces](https://wiki.bwhpc.de/e/Helix/Filesystems)
- [Kimi K2 Thinking vLLM recipe](https://recipes.vllm.ai/moonshotai/Kimi-K2-Thinking)
- [Kimi K2.5 vLLM recipe](https://recipes.vllm.ai/moonshotai/Kimi-K2.5)
- [Kimi K2.6 vLLM recipe](https://recipes.vllm.ai/moonshotai/Kimi-K2.6)
- [Kimi K3 vLLM recipe](https://recipes.vllm.ai/moonshotai/Kimi-K3)
- [Kimi K3 checkpoint card](https://huggingface.co/moonshotai/Kimi-K3)
- [Qwen3.8 Flash Next checkpoint card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Qwen3.8 27B checkpoint card](https://huggingface.co/Qwen/Qwen3.8-27B)
