# One-command OpenRouter completion workflow

This workflow lets a third user complete the six remaining OpenRouter models
from a clean clone. The user supplies only a Slurm partition, an OpenRouter
credential in a non-reserved process environment variable, and the name of
that credential variable. The controller creates its virtual environment, installs the
project, downloads the complete pinned HSLE Hugging Face dataset, authenticates
all inputs and live routes, submits the arrays, and writes structured results.

The six models are Kimi K2 Thinking, Kimi 2.5, Kimi 2.6, Kimi K3, Qwen 3.8
Max, and MiniMax M2.5.

## Run it

From the root of a clean clone:

```bash
export MY_OPENROUTER_KEY='...'
export HSLE_OPENROUTER_KEY_ENV=MY_OPENROUTER_KEY
export HSLE_SLURM_PARTITION=YOUR_PARTITION
bash script/run_remaining_openrouter_benchmarks.sh
```

The same controller can be scheduled:

```bash
sbatch \
  --partition "$HSLE_SLURM_PARTITION" \
  --export="PATH,HOME,XDG_CACHE_HOME,HSLE_OPENROUTER_KEY_ENV,HSLE_SLURM_PARTITION,${HSLE_OPENROUTER_KEY_ENV}" \
  script/run_remaining_openrouter_benchmarks.sh
```

Running it with `bash` on the login host is the simpler recommended form; the
controller itself performs only setup, validation, and `sbatch` submission.

For compatibility, the explicit two-argument form is also accepted. The first
argument is the credential variable's name, never its value:

```bash
bash script/run_remaining_openrouter_benchmarks.sh \
  MY_OPENROUTER_KEY YOUR_PARTITION
```

No Hugging Face credential is needed because the HSLE dataset is public.

## What the controller does

1. Validates the partition, non-reserved credential-variable name, paths, and
   Slurm tools. Hugging Face names, controller exports, `PATH`, `HOME`, cache
   control, and `SLURM_*`/`SBATCH_*` names are rejected as key-name collisions.
2. Creates `.venv-openrouter-benchmark/` and installs the checked-out project.
3. Downloads the complete public dataset
   `shashankskagnihotri/humanitys-second-last-exam` at immutable revision
   `aeda08b2536a19e698d027fd4f701eea78c9171d`.
4. Requires the exact 274-file revision inventory and validates the complete
   491-row, 174-column consolidated dataset, processed tables, corrections, and
   referenced image bundle. The consolidated dataset SHA-256 is
   `8a0568576ed1788e21899171c4e5a379b814ac09d912a26e3ad8a835a0337b04`.
5. Extracts the v3 single-dispatch input archive only after requiring archive
   SHA-256
   `ced06f31b7d82a58db28391f6e9bf09293a88933480f6b5354784ce98d3ede5f`.
6. Authenticates all 491 targets, 982 linked examples, 258 images, corrections,
   task partitions, prompt envelopes, and paid/no-replay exclusions.
7. Uses unpriced OpenRouter GET requests to authenticate the key and require
   every exact provider endpoint, price, context limit, output limit, and
   parameter contract before any `sbatch` call. Free-tier, management, and
   provisioning keys are rejected; an ordinary paid inference key is required.
   When the inference key has a numeric spending limit, at least **$1,501.72**
   must remain under that limit.
8. Stages a held release gate, one bounded Slurm array for each of the six
   routes, and a dependent finalizer. Only after every submission succeeds does
   it release the gate; a submission or release failure cancels the exact
   staged job IDs before any route can make a paid call.

Each worker repeats the unpriced key, canonical-alias, modality, endpoint, and
price checks for its route (and Gemini feedback when that shard needs LFE)
immediately before work. Kimi 2.5's expiration date is also checked before
every paid dispatch, so time spent queued cannot turn known expiry into scored
model failures.

By default, the complete dataset and input cache live below
`${HOME}/.cache/hsle/`; only generated work is written below the clone.

## Exactly one dispatch

Every benchmark-model generation turn has `max_retries=0` and
`max_total_attempts=1`. There is no prompt compaction or second attempt. The
first blank, malformed, rejected, transport-ambiguous, or otherwise unusable
model outcome becomes a terminal incorrect response with closeness 0. A
durable request intent without a trustworthy settlement is never replayed.

An LFE coordinate has three benchmark-generation turns: linked example 1,
linked example 2, and the target. After each linked example, Gemini 3.5 Flash
is called once through the same OpenRouter credential to create the binary
feedback shown to the evaluated model. A failed first feedback call records the
coordinate as `feedback_failed`: operationally incomplete and unjudged, with no
HLE or closeness score assigned to the evaluated model. It cannot trigger a
second generation or judge dispatch, and the strict finalizer exits nonzero so
the run cannot be mislabeled complete.

The feedback prompt and canonical response schema are byte-identical to the
partial campaign: prompt SHA-256
`0f0023ee579b8c134f1834ed8952778b9e01460e31d47c242ee3629da9d44835` and
canonical response-schema SHA-256
`aa7c81fc7de7e05210a632f975b821107c44b7184e9520eb359dace5dbe1c20e`.
Both hashes are enforced and recorded in requests, decisions, prepare evidence,
and the final manifest.

The v3 workload contains 10,295 callable coordinates, 14,719 benchmark
generation turns, and 4,424 LFE feedback calls. The 73 prior paid/no-replay
coordinates are authenticated but excluded from submission.

## Exact routes

The controller fails closed if any selected endpoint is missing, unhealthy,
repriced, or cannot enforce the required parameters.

| Model request | Required endpoint | Input/output price per 1M tokens | Reasoning |
|---|---|---:|---|
| `moonshotai/kimi-k2-thinking` | `novita/bf16` | $0.60 / $2.50 | native |
| `moonshotai/kimi-k2.5` | `deepinfra/fp4` | $0.45 / $2.25 | native |
| `moonshotai/kimi-k2.6` | `deepinfra/fp4` | $0.75 / $3.50 | native |
| `moonshotai/kimi-k3` | `sail-research/fp4` | $2.60 / $13.00 | `max` |
| `qwen/qwen3.8-max` | `alibaba` | $2.00 / $6.00 | `xhigh` |
| `minimax/minimax-m2.5` | `novita/fp8` | $0.30 / $1.20 | native |
| `google/gemini-3.5-flash` LFE feedback | `google-ai-studio/flex` | $0.75 / $4.50 | `low` |

All payloads use `provider.only`, an exact provider order,
`allow_fallbacks=false`, `require_parameters=true`, and a pinned maximum price.
Public model IDs are sent because the dated canonical aliases are not callable
lookup IDs; returned dated aliases are accepted and recorded.

The controller allows four active shards for each Novita, DeepInfra, and
Alibaba route, while the new Sail Research Kimi K3 route stays at one. Thus the
two shared Novita routes are capped at eight concurrent workers in total and so
are the two shared DeepInfra routes; Alibaba is capped at four and Sail at one.
Historical campaign evidence supports the Novita, DeepInfra, and Alibaba caps
while giving Kimi 2.5 enough throughput to finish before expiration. Sail
Research is a new endpoint with no historical concurrency evidence here, so
Kimi K3 is deliberately serialized at one active worker.
Provider-specific generation admissions are spaced before durable intent
publication. Gemini feedback admissions use one shared one-second gate across
all workers, so waiting for capacity cannot create a false no-replay settlement.

An ordinary inference key cannot query OpenRouter's account-wide `/credits`
endpoint; that endpoint requires a separate management key. The controller
therefore cannot prove the account's cash balance under the promised one-key
interface. Before launch, the third user must fund at least the planning
reserve shown in the cost audit or enable sufficient automatic funding. A
numeric per-key limit is enforced, but an unlimited key does not prove the
underlying shared account balance.

Kimi 2.5 has a model-wide OpenRouter expiration date of **31 August 2026**.
The workload must finish before then. Preflight refuses to launch on or after
that UTC date, even if a stale endpoint listing still appears healthy.

See [the cost audit](OPENROUTER_COSTS.md) for the current planning scenarios
and why historical OpenAI and Claude runs were much cheaper.

## Structured output

The default output is the Git-ignored directory:

```text
need_to_be_judged/
├── control/                 # authenticated prepare and worker summaries
├── generation_results/      # one signed scientific result per coordinate
│   └── ROUTE/EVALUATION_KEY.json
├── lfe_feedback_api/        # raw single-dispatch Gemini feedback settlements
├── lfe_feedback_decisions/  # signed binary feedback used in LFE prompts
├── lfe_feedback_requests/   # signed requests binding question/answer/response
├── raw_api/                 # durable intents and provider settlements
├── logs/                    # operational Slurm stdout/stderr; not hash-inventoried
└── RUN_MANIFEST.json        # final authenticated inventory and completion census
```

Set `HSLE_OUTPUT_ROOT` to an absolute path before launch to place the entire
tree elsewhere. It must not equal or contain the clone and must not overlap the
dataset cache, authenticated input cache, or virtual environment. The one
permitted child of the clone is the default ignored `need_to_be_judged/`
directory; other custom locations must be outside the clone. The final HLE
correctness and 0–10 closeness judgments are not run on the third user's
OpenRouter account; the returned generation results are explicitly marked as
requiring Shashank Agnihotri's final Gemini judging step.

The OpenRouter key value is never accepted as a command-line argument or
written to disk. Installation and Hugging Face download run with provider
credentials removed from their environments. Prepare receives only the named
OpenRouter credential and, when explicitly requested for upstream validation,
`HF_TOKEN`. Worker jobs receive only the one named OpenRouter credential plus
explicit non-secret paths, and the finalizer receives no credential.

`RUN_MANIFEST.json` hashes every signed scientific/control/API artifact. Slurm
logs are intentionally excluded because the scheduler can append or flush the
finalizer's own stdout/stderr after the manifest is written; they remain in the
structured tree as operational diagnostics.

## Restart behavior

Rerunning the same command after every previously submitted job ID is terminal
is safe. Do not start a second controller against the same output directory
while an earlier array or finalizer is active. Signed successful results are
validated and skipped. Any existing durable intent without a valid settlement
becomes a paid ambiguity and is not replayed. The finalizer succeeds only when
all six arrays produced a scientifically terminal model result for every one
of the 10,295 callable coordinates and no LFE feedback call remains
operationally incomplete. It also requires the exact top-level output topology;
otherwise it writes no completion manifest and exits nonzero.

This workflow makes real paid provider calls. It has no smoke-test or trial-call
mode in the documented path.
