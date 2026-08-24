# Public OpenRouter stopped-route resume

This release helper covers exactly five unfinished OpenRouter reproduction
routes: Kimi K2 Thinking, Kimi 2.5, Kimi 2.6, Kimi K3, and the project's exact
Qwen OpenRouter route `qwen/qwen3.8-max` (requested as the pinned
`qwen/qwen3.8-max-20260803` Alibaba route). It does not call a judge.

Kimi 2.6 is requested through the working canonical OpenRouter slug
`moonshotai/kimi-k2.6`. The earlier dated
`moonshotai/kimi-k2.6-20260420` request is retained only as failed 404 lineage
in the signed input and run manifests; the public runner never dispatches it.

## Frozen input download

The scientific inputs are deliberately not duplicated in this Git repository.
On the first run, the controller downloads without authentication
`openrouter/hsle_public_openrouter_resume_v2.tar.gz` from the public Hugging
Face dataset `shashankskagnihotri/humanitys-second-last-exam` at immutable
revision `582207e5bd95b4f4e2948887c2d613398e98a17e`. No Hugging Face token is
needed. The archive must also have SHA-256
`226ba161608182f37ee4310bd8d3cb32457604603f272ad3a88ee5d0666ecd23`;
unsafe archive entries or any mismatch fail closed. Extraction is atomic into:

```text
${XDG_CACHE_HOME}/hsle/public-openrouter-resume-v2
```

When `XDG_CACHE_HOME` is not an absolute path, the cache is
`${HOME}/.cache/hsle/public-openrouter-resume-v2`. Subsequent runs reuse the
authenticated tree and do not redownload it. The dataset revision and archive
SHA-256 are release constants and cannot be overridden at runtime.

An authorized pre-populated tree remains supported and takes precedence over
the automatic download. Set its absolute path when required:

```bash
export HSLE_INPUT_ROOT=/absolute/path/to/authorized-hsle-inputs
```

Resolution order is explicit `HSLE_INPUT_ROOT`, an already-warm user cache, the
ignored project-local `.hsle_public_resume_inputs_v2/` directory, and finally
the verified Hugging Face download into the user cache.

Before creating `needs_to_be_judged/` or invoking `sbatch`, preflight requires:

- the exact 491 target rows and 982 unique, two-per-target contexts;
- the pinned linkage and correction layer, including the exact per-field
  correction counts;
- exactly 258 referenced image files with no missing, extra, or symlinked
  inventory entries;
- the exact five route task counts and callable-vector hashes;
- pairwise-disjoint settled, paid/no-replay, and callable sets whose union is
  the full canonical route universe; and
- the release-pinned provider-visible prompt/source hashes, including ordered
  image-byte hashes.

`INPUT_MANIFEST.json` itself must have SHA-256
`fc1e7f134626df95c2092ca07c5f3932a47b59ea8660633202e307e5cdb508dd`.
Any mismatch fails closed before an output directory or scheduler submission.
The custom context/linkage and correction layer cannot be reconstructed from
the official HLE dataset alone.

Version 2 contains 10,127 safely callable coordinates and separately binds 60
paid/no-replay coordinates. It supersedes version 1 after a raw-WAL audit
reclassified 29 K3/Qwen rows: 17 are callable and 12 are paid/no-replay. No
question, example, correction, or image byte changed.

Immediately before submission, the controller also performs six unpriced GET
requests: one authenticated key check and one public endpoint-catalog check per
route. Every exact provider tag must be uniquely present with status `0`, the
pinned provider, price, context/output limits, and required parameters. No chat
completion is called, and the credential and raw account response are not
persisted. A degraded exact route therefore stops the launch before `sbatch`.

Optional official-source access validation is available without changing the
two-argument interface:

```bash
export HSLE_VALIDATE_OFFICIAL_HF=1
export HF_TOKEN='...'
```

This separately validates access to `cais/hle` at pinned revision
`5a81a4c7271a2a2a312b9a690f0c2fde837e4c29`. Only the fixed name `HF_TOKEN`
is accepted. Its value is neither persisted nor exported to workers or the
finalizer. This optional validation credential is not used for the public HSLE
archive download.

## Run

From a clean clone, export an OpenRouter key under any environment-variable
name and pass that *name*, not the secret value, with the Slurm partition:

```bash
export MY_OPENROUTER_KEY='...'
bash script/run_remaining_openrouter_benchmarks.sh MY_OPENROUTER_KEY PARTITION
```

The controller can itself be scheduled; the partition argument is applied to
all child arrays. Run this form from the clean clone root so Slurm's immutable
`SLURM_SUBMIT_DIR` resolves the repository even though it executes a spool copy:

```bash
cd /absolute/path/to/Humanitys_Second_Last_Exam
sbatch script/run_remaining_openrouter_benchmarks.sh MY_OPENROUTER_KEY PARTITION
```

The one entry point has exactly those two positional arguments. It creates
`.venv-public-openrouter-resume`, installs the project and API dependencies,
downloads or locates and then authenticates the frozen input tree, and submits
one bounded array per route. It writes only to `needs_to_be_judged/`. The key
value is never an argument, log field, manifest field, or project file. The
bootstrap, package installation, and Hugging Face download processes do not
receive it. The input root's absolute path is likewise omitted from
transferable records.

Static results finish directly. Exact learning-from-experience generation
cannot scientifically continue from example 1 to example 2 without the
Gemini 3.5 Flash correctness feedback that becomes part of the next prompt.
Therefore the public runner stops at each feedback boundary and writes bound
requests under `lfe_feedback_requests/`. Share the whole
`needs_to_be_judged/` directory with the project owner. After the owner runs
Gemini 3.5 Flash and returns matching files under
`lfe_feedback_decisions/`, rerun the same one-line command. It advances to the
next boundary without replaying settled OpenRouter calls. Two such feedback
round trips complete each LFE target response.

The transferable folder contains:

- `generation_results/`: scientific generation records ready for later HLE
  and closeness judging;
- `raw_api/`: durable, secret-free dispatch intents, raw provider responses,
  receipts, rejections, and explicit paid ambiguities;
- `lfe_feedback_requests/` and `lfe_feedback_decisions/`: cryptographically
  bound feedback relay files;
- `RUN_MANIFEST.json`: hashes, exact route identities, and completion status.

An intent without a definitive settlement is never automatically replayed.
The manifest records that coordinate as paid-ambiguous for manual resolution.
Neither LFE feedback judging nor final HLE/closeness judging is implemented in
the public runner.
