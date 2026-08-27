# Humanity's Second Last Exam

**Creator, author, and project owner: [Shashank Agnihotri](https://github.com/shashankskagnihotri)**

Humanity's Second Last Exam (HSLE) measures how difficult-question answering
changes when a model receives zero examples, one example, two examples, or
feedback derived from prior experience.

This is Shashank Agnihotri's clean software repository. It contains benchmark,
judging, aggregation, plotting, and pinned local-cluster generation code. It
does not contain API credentials, model responses, judge responses, metrics,
plots, logs, cluster reports, or debugging artifacts.

The reconciled 27 August 2026 generation scope contains **33 models**, **4
named settings**, and **77,685 concrete prompt coordinates**. Twenty-four
models are complete across zero-shot, one-shot A, one-shot B, two-shot, and
learning from experience. There are 63,443 real answers, 449 terminal
incorrect settlements, 13,725 callable coordinates, and 68 paid/no-replay
coordinates. See [the complete model-by-setting census](docs/BENCHMARK_STATUS.md).

The current scored plot release contains **25 models**: the 24
generation-complete models plus MiniMax M2.5 under an explicit conservative
lower-bound policy. It consists of exactly **26 one-page vector PDFs**: two
all-model plots and HLE-accuracy/closeness pairs for each of the 12 represented
families. Unfinished generation models are not presented as complete and are
not included in those figures.

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

The scope has 24 multimodal and 9 text-only models across 13 families. Stable
model identifiers, modalities, ordering, and audited generation partitions are
in [`configs/models.yaml`](configs/models.yaml).

The newest study entry is the exact multimodal
`Qwen/Qwen3.8-Flash-Next` checkpoint at revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`, using its official default
`reasoning_effort=xhigh`. Its 2,455 coordinates are currently unresolved and
are not counted among the 24 complete models.

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
- authenticated provenance and audit metadata for the consolidated release.

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

## Helix local H200 workflow

The active collaborator handoff serves capacity-compatible pinned checkpoints
locally on one Helix `gpu-single` node with eight H200 GPUs. A launchable entry
command locates or creates a 10-TB workspace, submits a CPU preparation job,
and submits one dependent exclusive GPU generation job. Preparation creates
the virtual environment and downloads the pinned model snapshot and pinned
public HSLE dataset; no compute runs on the login node.

Provide Gemini 3.5 Flash only through a protected key file for inline binary
learning-from-experience feedback. The pinned public model and dataset
snapshots use public unauthenticated downloads, and the submitted jobs inherit
no login-shell credentials:

```bash
export HSLE_GEMINI_KEY_FILE=/secure/path/gemini.key

bash script/run_helix_kimi_k26.sh
bash script/run_helix_kimi_k25.sh
bash script/run_helix_kimi_k2_thinking.sh
bash script/run_helix_qwen38_27b.sh
```

Each GPU job cryptographically verifies every downloaded checkpoint shard and
the model/tokenizer configuration against its signed preparation authority
before starting the local server.

The commands cover the complete pinned coordinate universe for their model:
2,455 coordinates for each multimodal checkpoint and 2,085 text-only
coordinates for Kimi K2 Thinking. Coordinate concurrency defaults to 16 and
can be changed with `HSLE_HELIX_CONCURRENCY`; every coordinate preserves its
internal LFE order.

Every model or feedback call has an immutable write-ahead intent and at most
one request attempt. A blank, malformed, or failed first model response is a
terminal incorrect result. An interrupted intent is never replayed, while a
coordinate with no intent can resume safely. Structured JSON and an
import-ready CSV are written below the workspace's
`need_to_be_judged/<route>/` directory. Final HLE correctness and closeness
judging is deliberately deferred to the project owner.

Kimi K3 requires a guarded distributed handoff and fails before workspace
creation or job submission on the one-node path:

```bash
bash script/run_helix_kimi_k3.sh
```

The official Kimi K3 vLLM recipe needs about 1,680 GB of GPU memory, more than
one eight-H200 Helix node provides, and its supported runtime is a CUDA 13
container/nightly multi-node path rather than the pip-wheel CPU-offload path.
K3 remains blocked until a documented 16-GPU Helix allocation and that exact
official recipe are implemented.

The Qwen entry serves the exact native-multimodal `Qwen/Qwen3.8-27B` checkpoint
at immutable revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; it is a new
scientific identity and does not inherit evidence from the former hosted Max
route. See the [complete Helix contract](docs/HELIX_LOCAL.md).

## General reproduction pipeline

Generated artifacts remain below the ignored `outputs/` tree.

Generate responses:

```bash
python -m hsle.benchmark \
  --provider PROVIDER \
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
├── docs/                       # Status, Helix, prompts, corrections, credentials
├── script/
│   ├── run_helix_kimi_*.sh     # Four identity entries; K3 fails closed
│   ├── run_helix_qwen38_27b.sh # Pinned local Qwen3.8 27B entry command
│   └── workers/helix_*.sh      # CPU preparation and GPU generation
├── src/hsle/                   # Benchmark, judge, metrics, plots, runner
├── data/                       # Downloaded dataset; ignored
├── outputs/                    # Generated artifacts; ignored
└── need_to_be_judged/          # Import-ready generated results; ignored
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
