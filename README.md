# Humanity's Second Last Exam

**Author and project owner: [Shashank Agnihotri](https://github.com/shashankskagnihotri)**

Humanity's Second Last Exam (HSLE) measures how difficult-question answering
changes when a model receives zero examples, one example, two examples, or
feedback derived from prior experience.

This repository is the clean software release. It contains benchmark, judging,
aggregation, plotting, and third-cluster resume code. It does **not** contain
API credentials, benchmark responses, judge responses, metrics, plots, logs,
or cluster debugging artifacts.

The audited 24 August 2026 generation snapshot covers 33 models and 77,315
expected prompt coordinates. Twenty-four models are complete across every
concrete variant; 13,148 coordinates remain scientifically eligible for a
fresh call if their route, credit, cluster, and owner authorization are
available, while 73 paid requests are unresolved but must not be replayed. See
[the complete benchmark census](docs/BENCHMARK_STATUS.md) for every model and
setting. The final 33-model judged matrix and final plots are not yet assembled.

## Evaluation design

HSLE has four named settings and five concrete response variants because
one-shot is repeated with two prespecified example assignments.

| Named setting | Concrete variant(s) | Context supplied |
|---|---|---|
| Zero-shot | `zero_shot` | Target question only |
| One-shot | `one_shot_a`, `one_shot_b` | Linked example A or B |
| Two-shot | `two_shot` | Both linked examples |
| Learning from experience | `learning_from_experience` | Both examples, with feedback from prior example performance |

The target set contains 491 questions: 417 text-only questions and 74 that
require an image. Multimodal models therefore contribute 2,455 coordinates
when complete; text-only models contribute 2,085.

```text
multimodal: 491 × (1 zero + 2 one-shot + 1 two-shot + 1 LFE) = 2,455
text-only:  417 × (1 zero + 2 one-shot + 1 two-shot + 1 LFE) = 2,085
```

The intended scope has 23 multimodal and 10 text-only models across 13
families. The authoritative identifiers, modalities, display order, and
audited generation status are in [`configs/models.yaml`](configs/models.yaml).

The one-shot score is paired within each model-question unit before
aggregation:

```text
one_shot(model, question) =
    (one_shot_a(model, question) + one_shot_b(model, question)) / 2
```

Zero-shot, paired one-shot, two-shot, and learning from experience are then
the four equally weighted named settings.

## Dataset

The public data release is
[`shashankskagnihotri/humanitys-second-last-exam`](https://huggingface.co/datasets/shashankskagnihotri/humanitys-second-last-exam),
pinned by default at immutable revision
`6861ef237eb9501b8fda3d4fe61788154e143c22`.

It contains:

- one primary **491-row × 174-column consolidated `test` split**, with the
  complete 58-field final target, instance A, and instance B records on every
  row;
- 491 corrected target rows;
- 982 unique context-example rows, exactly two per target;
- deterministic target-to-example linkages;
- 258 referenced image files;
- target and context correction manifests; and
- the authenticated five-route OpenRouter resume archive.

Download and validate the default snapshot:

```bash
hsle-download-data
```

Or select an explicit public revision:

```bash
hsle-download-data \
  --repo-id shashankskagnihotri/humanitys-second-last-exam \
  --revision 6861ef237eb9501b8fda3d4fe61788154e143c22
```

The downloader writes only to the Git-ignored `data/` directory and validates
recorded hashes, exact row counts, correction manifests, and the complete
image-reference set. Public download requires no authentication; `HF_TOKEN` is
only needed for separately gated upstream data or model weights.

The processed release applies explicit corrections rather than silently
overwriting provenance:

- 22 targets have a corrected question component;
- 93 have a corrected answer component, including 91 semantic changes;
- 160 have a corrected rationale component;
- 172 targets have at least one corrected component; and
- one linked context example has corrected question, answer, and rationale.

See [correction provenance](docs/CORRECTIONS.md) and the dataset card for the
component-level manifests and source revisions.

## Installation

```bash
git clone https://github.com/shashankskagnihotri/Humanitys_Second_Last_Exam.git
cd Humanitys_Second_Last_Exam
conda env create --file environment.yml
conda activate hsle
python -m pip install --editable .
hsle-download-data
```

The package requires Python 3.11 or newer. Provider credentials belong in the
process environment or an untracked `.env`; see
[credential configuration](docs/CREDENTIALS.md).

## Resume the five OpenRouter routes on a third cluster

The public runner covers exactly:

- Kimi K2 Thinking;
- Kimi 2.5;
- Kimi 2.6;
- Kimi K3; and
- Qwen 3.8 Max.

It does not cover the 2,085 Kimi K2 Instruct coordinates, the 187 MiniMax
coordinates currently held by the project spending cap, or the 749 remaining
local Nemotron coordinates. Those counts are part of the full census but
require their separate route or renewed authority.

From a clean clone, export the third user's OpenRouter key under any valid
environment-variable name and pass the **name**, never the key value, together
with the Slurm partition:

```bash
export MY_OPENROUTER_KEY='...'
bash script/run_remaining_openrouter_benchmarks.sh MY_OPENROUTER_KEY PARTITION
```

The controller may itself be scheduled from the clone root:

```bash
sbatch script/run_remaining_openrouter_benchmarks.sh MY_OPENROUTER_KEY PARTITION
```

On first use the runner downloads without authentication
`openrouter/hsle_public_openrouter_resume_v2.tar.gz` from the pinned dataset
revision. It requires archive SHA-256
`226ba161608182f37ee4310bd8d3cb32457604603f272ad3a88ee5d0666ecd23`,
extracts it safely and atomically, and validates the full scientific input,
route partitions, task vectors, prompt envelopes, image bytes, and live exact
provider routes before any Slurm submission.

The archive contains 10,127 safely callable coordinates and separately binds
60 paid/no-replay coordinates. Settled and ambiguous paid work is never
automatically replayed. If an exact provider route is unhealthy—Kimi K3's
Morph route was degraded at the release audit—the controller fails closed
before submitting any array.

Static coordinates can finish in one pass. LFE coordinates stop at each
Gemini-feedback boundary and write a bound request under
`needs_to_be_judged/lfe_feedback_requests/`. After the project owner returns
the matching decision under `lfe_feedback_decisions/`, rerun the same command;
the coordinate advances without replaying settled OpenRouter calls.

The runner writes only to the Git-ignored `needs_to_be_judged/` tree. It does
not run final HLE or closeness judging. The complete recovery, transfer, and
no-replay contract is documented in
[the OpenRouter handoff guide](docs/OPENROUTER_RESUME.md).

## General reproduction pipeline

All generated artifacts remain beneath the ignored `outputs/` tree.

### Generate responses

Choose a provider-facing model name separately from the stable analytical
identifier in `configs/models.yaml`:

```bash
python -m hsle.benchmark \
  --provider openrouter \
  --model provider/model-name \
  --model-id ANALYTICAL_MODEL_ID \
  --setting zero_shot \
  --output outputs/responses/model_responses.csv
```

Run the command once for each named setting. A `one_shot` invocation produces
both A and B variants. Use `--dry-run` to materialize and inspect prompts
without making a provider call.

### Judge HLE correctness and closeness

```bash
python -m hsle.judge \
  --metric both \
  --input outputs/responses/model_responses.csv \
  --judge-model gemini-3.5-flash \
  --output outputs/judgments/judged_responses.csv
```

HLE correctness uses the pinned official HLE evaluation prompt. Closeness is a
separate project-defined integer rubric from 0 through 10. Their exact prompts,
schemas, and hashes are documented in [prompt protocols](docs/PROMPTS.md).

### Aggregate only a complete judged matrix

```bash
python -m hsle.metrics \
  --input outputs/judgments/judged_responses.csv \
  --originals data/processed/hsle_original_questions.csv \
  --missing-policy none \
  --metrics-output outputs/metrics/model_setting_metrics.csv \
  --std-output outputs/metrics/model_setting_closeness_std.csv
```

The aggregator rejects duplicate, unexpected, or missing coordinates. It does
not synthesize scores for generation failures or missing judge rows. Every
judged row must declare `generation_completion_type=real_nonblank` or
`terminal_policy`; real rows require a nonblank response, while terminal rows
require blank response fields, HLE incorrect, and closeness 0.

### Render figures

```bash
python -m hsle.plotting \
  --metrics outputs/metrics/model_setting_metrics.csv \
  --closeness-std outputs/metrics/model_setting_closeness_std.csv \
  --output-dir outputs/figures
```

For a complete 33-model matrix, the renderer produces 28 vector PDFs: two
all-model figures and two figures for each of 13 families, plus a SHA-256
manifest. Plotting validates the complete model-setting grid and the 491/417
denominators before writing output.

## Repository layout

```text
.
├── configs/                    # Model, provider, and plotting contracts
├── docs/                       # Status, prompts, corrections, credentials, handoff
├── script/
│   ├── run_remaining_openrouter_benchmarks.sh
│   └── workers/run_public_openrouter_resume_shard.sh
├── src/hsle/                   # Benchmark, judge, metrics, plots, resume engine
├── data/                       # Downloaded dataset; ignored
├── outputs/                    # Generated replication artifacts; ignored
└── needs_to_be_judged/         # Third-cluster handoff; ignored
```

No data, response, judgment, metric, plot, log, virtual environment, or cache
file is tracked by Git.

## Reproducibility boundary

This release supports method auditing and new end-to-end replication. Hosted
model aliases, provider implementations, pricing, and availability can change;
a later run under the same protocol is not necessarily a byte-identical
reconstruction of a historical API response. Record provider-returned model
identifiers, execution timestamps, prompt hashes, and generation parameters
for every new run.

The current status report is a generation census, not a claim of final paper
results. A final numeric comparison must wait for the remaining generation,
the authoritative 33-model dual-judge matrix, and successful plot assembly.

## License

The software is licensed under the
[GNU General Public License v3.0](LICENSE) (`GPL-3.0-only`). The code license
does not grant additional rights to separately hosted benchmark data,
third-party images, model weights, or provider outputs.

Copyright © 2026 Shashank Agnihotri. See [AUTHORS.md](AUTHORS.md) and
[CITATION.cff](CITATION.cff) for attribution and citation metadata.
