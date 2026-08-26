# Humanity's Second Last Exam

**Creator, author, and project owner: [Shashank Agnihotri](https://github.com/shashankskagnihotri)**

Humanity's Second Last Exam (HSLE) measures how difficult-question answering
changes when a model receives zero examples, one example, two examples, or
feedback derived from prior experience.

This is Shashank Agnihotri's clean software repository. It contains benchmark,
judging, aggregation, plotting, and portable OpenRouter completion code. It
does not contain API credentials, model responses, judge responses, metrics,
plots, logs, cluster reports, or debugging artifacts.

The reconciled 25 August 2026 generation scope contains **32 models**, **4
named settings**, and **75,230 concrete prompt coordinates**. Twenty-four
models are complete across zero-shot, one-shot A, one-shot B, two-shot, and
learning from experience. There are 63,764 real answers, 533 terminal
incorrect settlements, 10,860 callable coordinates, and 73 paid/no-replay
coordinates. See [the complete model-by-setting census](docs/BENCHMARK_STATUS.md).

The current scored plot release contains **25 models**: the 24
generation-complete models plus MiniMax M2.5 under an explicit conservative
lower-bound policy. It consists of exactly **26 one-page vector PDFs**: two
all-model plots and HLE-accuracy/closeness pairs for each of the 12 represented
families. The five pending OpenRouter models and both unfinished Nemotron
models are not presented as complete and are not included in those figures.

## Evaluation design

Four named settings produce five concrete response variants because one-shot
uses two fixed example assignments.

| Named setting | Concrete variant(s) | Context supplied |
|---|---|---|
| Zero-shot | `zero_shot` | Target question only |
| One-shot | `one_shot_a`, `one_shot_b` | Linked example A or B |
| Two-shot | `two_shot` | Both linked examples |
| Learning from experience | `learning_from_experience` | Both examples plus feedback on prior example performance |

The target set has 491 questions: 417 text-only and 74 image-dependent.
Multimodal models therefore contribute 2,455 coordinates when complete;
text-only models contribute 2,085.

```text
multimodal: 491 × (1 zero + 2 one-shot + 1 two-shot + 1 LFE) = 2,455
text-only:  417 × (1 zero + 2 one-shot + 1 two-shot + 1 LFE) = 2,085
```

The scope has 23 multimodal and 9 text-only models across 13 families. Stable
model identifiers, modalities, ordering, and audited generation partitions are
in [`configs/models.yaml`](configs/models.yaml).

One-shot is paired inside each model-question unit:

```text
one_shot(model, question) =
    (one_shot_a(model, question) + one_shot_b(model, question)) / 2
```

Zero-shot, paired one-shot, two-shot, and LFE are then the four equally
weighted named settings.

## Complete Hugging Face dataset

The public dataset is
[`shashankskagnihotri/humanitys-second-last-exam`](https://huggingface.co/datasets/shashankskagnihotri/humanitys-second-last-exam),
pinned by default at immutable revision
`aeda08b2536a19e698d027fd4f701eea78c9171d`.

Its primary `test` split is one **491-row × 174-column consolidated dataset**.
Every row contains the complete 58-field final target record, complete
58-field instance A record, and complete 58-field instance B record. The
release also contains:

- 491 corrected final targets;
- 982 unique linked context instances, exactly two per target;
- target-to-instance linkages and full provenance columns;
- 258 referenced image files;
- target and context correction manifests; and
- the authenticated historical six-route OpenRouter single-dispatch input bundle.

Download and validate the complete default snapshot:

```bash
hsle-download-data
```

Or pin it explicitly:

```bash
hsle-download-data \
  --repo-id shashankskagnihotri/humanitys-second-last-exam \
  --revision aeda08b2536a19e698d027fd4f701eea78c9171d
```

The downloader requires the consolidated table to contain exactly 491 rows and
174 columns with SHA-256
`8a0568576ed1788e21899171c4e5a379b814ac09d912a26e3ad8a835a0337b04`.
It also validates processed tables, corrections, linkages, and the complete
image-reference set. The public release requires no Hugging Face credential.

The processed release records explicit corrections rather than hiding them:

- 22 targets have a corrected question component;
- 93 have a corrected answer component, including 91 semantic changes;
- 160 have a corrected rationale component;
- 172 targets have at least one corrected component; and
- one linked context instance has corrected question, answer, and rationale.

See [correction provenance](docs/CORRECTIONS.md) for the component-level audit.

## Installation

```bash
git clone https://github.com/shashankskagnihotri/Humanitys_Second_Last_Exam.git
cd Humanitys_Second_Last_Exam
conda env create --file environment.yml
conda activate hsle
python -m pip install --editable .
hsle-download-data
```

Python 3.11 or newer is required. Credentials belong only in the process
environment or an untracked `.env`; see
[credential configuration](docs/CREDENTIALS.md).

## Finish the five remaining OpenRouter models on a third cluster

The portable workflow covers exactly Kimi K2 Thinking, Kimi 2.5, Kimi 2.6,
Kimi K3, and Qwen 3.8 Max. MiniMax M2.5's safely callable remainder completed
on 25 August 2026 and is deliberately excluded, so a fresh clone cannot
duplicate it. The third user sets a Slurm partition, stores their OpenRouter
credential in a non-reserved environment variable, and identifies that
variable by name:

```bash
export MY_OPENROUTER_KEY='...'
export HSLE_OPENROUTER_KEY_ENV=MY_OPENROUTER_KEY
export HSLE_SLURM_PARTITION=YOUR_PARTITION
bash script/run_remaining_openrouter_benchmarks.sh
```

The script creates and installs its own virtual environment, downloads the
complete pinned Hugging Face dataset, authenticates the v3 task bundle and
every live exact provider route, then submits all five Slurm arrays plus a
strict completion finalizer.

Every benchmark-model turn is dispatched **exactly once**. There are zero
automatic retries and no prompt compaction. A blank, malformed, rejected, or
ambiguous first model outcome is terminal incorrect with closeness 0. LFE
feedback is obtained automatically from Gemini 3.5 Flash through the same
OpenRouter credential, also with one dispatch. A failed feedback call is
recorded as operationally incomplete, receives no model score, and makes the
strict finalizer fail; it is never retried or charged against the evaluated
model. The default structured result directory is the Git-ignored
`need_to_be_judged/` tree.

The remaining workload contains 10,111 callable coordinates, 14,535 benchmark
generation turns, 4,424 LFE feedback calls, and 60 authenticated prior
paid/no-replay exclusions. Final HLE and closeness judging is not charged to
the third user's OpenRouter account.

See [the complete one-command contract](docs/OPENROUTER_RESUME.md) and
[the corrected cost audit](docs/OPENROUTER_COSTS.md). The historical-output
projection is about **$1,035.22 inference**; the main cost driver is native
reasoning-token volume, especially Kimi K3, not a general OpenRouter markup.
The immutable v3 input archive remains unchanged; the live runner explicitly
binds K3's vanished Sail endpoint to DeepInfra BF16. The completed MiniMax
route remains in the immutable archive only as authenticated historical input;
neither the launcher nor the worker CLI accepts it as pending work.

## General reproduction pipeline

Generated artifacts remain below the ignored `outputs/` tree.

Generate responses:

```bash
python -m hsle.benchmark \
  --provider openrouter \
  --model provider/model-name \
  --model-id ANALYTICAL_MODEL_ID \
  --setting zero_shot \
  --output outputs/responses/model_responses.csv
```

A `one_shot` invocation produces both fixed A and B variants.

Judge HLE correctness and closeness:

```bash
python -m hsle.judge \
  --metric both \
  --input outputs/responses/model_responses.csv \
  --judge-model gemini-3.5-flash \
  --output outputs/judgments/judged_responses.csv
```

Aggregate only a complete judged matrix:

```bash
python -m hsle.metrics \
  --input outputs/judgments/judged_responses.csv \
  --originals data/processed/hsle_original_questions.csv \
  --missing-policy none \
  --metrics-output outputs/metrics/model_setting_metrics.csv \
  --std-output outputs/metrics/model_setting_closeness_std.csv
```

The aggregator rejects duplicate, unexpected, or missing coordinates. It does
not invent scores for absent generation or judge rows. Real rows require a
nonblank response; authenticated terminal rows require HLE incorrect and
closeness 0.

Render the current exact 25-model figure release from its coordinate-level
scored layer:

```bash
python -m hsle.plotting \
  --input outputs/judgments/current_25_model_scored_coordinates.csv \
  --output-dir outputs/figures
```

The plotter validates the exact 59,155-coordinate scored universe before it
writes anything. The input must retain `concrete_variant`,
`original_question_id`, binary `hle_correct`, integer `closeness_score`, and
the three explicit `operational_missing`, `hle_metric_missing`, and
`closeness_metric_missing` flags. One-shot A/B values are paired by original
question. All-model plots use each model's native 491- or 417-question cohort;
a family containing any text-only model is compared on the shared 417 text
questions, while an all-multimodal family uses all 491 questions.

The output is exactly 26 PDFs and no PNG, SVG, manifest, title, subtitle,
footnote, or explanatory prose. Bars preserve the four setting colors and
hatches, model labels preserve the family color scheme, and the rendering
theme is Seaborn `talk` with `whitegrid`. Generated score layers and PDFs stay
under the ignored `outputs/` tree and are not committed to Git.

## Repository boundary

```text
.
├── configs/                    # Model, provider, and plotting contracts
├── docs/                       # Status, prompts, corrections, credentials, costs
├── script/
│   ├── run_remaining_openrouter_benchmarks.sh
│   └── workers/run_public_openrouter_resume_shard.sh
├── src/hsle/                   # Benchmark, judge, metrics, plots, runner
├── data/                       # Downloaded dataset; ignored
├── outputs/                    # Generated artifacts; ignored
└── need_to_be_judged/          # Third-cluster results; ignored
```

No dataset snapshot, response, judgment, metric, plot, log, virtual
environment, cache, test artifact, or debugging report is tracked by Git.

Hosted model aliases, provider implementations, pricing, and availability can
change. New runs therefore record exact provider-returned identities,
timestamps, prompt hashes, provider pins, and generation parameters.

## License and citation

Copyright © 2026 Shashank Agnihotri.

The GitHub software and the separately hosted HSLE dataset release are
published under the [GNU General Public License v3.0](LICENSE). See
[AUTHORS.md](AUTHORS.md) and [CITATION.cff](CITATION.cff) for Shashank
Agnihotri's attribution and citation metadata.
