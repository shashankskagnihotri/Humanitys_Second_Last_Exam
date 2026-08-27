# Credentials and provider selection

The repository contains credential variable names only. It does not contain
API keys or tokens. Runtime configuration accepts standard environment
variables from either an untracked `.env` file or the process environment.

## Create a local `.env`

From the repository root:

```bash
cp .env.example .env
```

Open `.env` and fill only the provider and dataset settings needed for the
intended run:

```dotenv
GEMINI_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
HF_TOKEN=
HUGGINGFACE_HUB_TOKEN=
OPENROUTER_API_KEY=
HSLE_GEMINI_KEY_FILE=

HSLE_DATASET_REPO=shashankskagnihotri/humanitys-second-last-exam
HSLE_DATASET_REVISION=aeda08b2536a19e698d027fd4f701eea78c9171d
OPENROUTER_SITE_URL=
OPENROUTER_APP_NAME=Humanitys-Second-Last-Exam
```

The package automatically reads `.env` from the repository root.
Already-exported process variables take precedence; `.env` does not overwrite
them. The checkout root can be overridden for automation with
`HSLE_PROJECT_ROOT`.

To export the same local file for commands outside the Python package:

```bash
set -a
source .env
set +a
```

## Standard variables

| Variable | Purpose | Required when |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API credential | Generating with the Gemini provider; running HLE or closeness judging; running learning from experience |
| `OPENAI_API_KEY` | OpenAI API credential | Generating with the OpenAI provider |
| `ANTHROPIC_API_KEY` | Anthropic API credential | Generating with the Anthropic provider |
| `HF_TOKEN` | Hugging Face access token | Accessing separately gated upstream data or model weights; not needed for the public HSLE release |
| `HUGGINGFACE_HUB_TOKEN` | Alternative Hugging Face access-token name | Generic authenticated upstream access; ignored by the public unauthenticated Helix preparation workflow |
| `OPENROUTER_API_KEY` | OpenRouter API credential | Generating through OpenRouter |
| `HSLE_GEMINI_KEY_FILE` | Path to a protected file containing one Gemini key | Helix inline binary LFE feedback |
| `HSLE_DATASET_REPO` | Hugging Face dataset repository identifier | Optional override of the pinned public default |
| `HSLE_DATASET_REVISION` | Immutable dataset revision | Optional override of the pinned public default |
| `OPENROUTER_SITE_URL` | Optional OpenRouter attribution URL | Supplying the optional `HTTP-Referer` header |
| `OPENROUTER_APP_NAME` | Optional OpenRouter application label | Supplying the optional `X-Title` header |

The exact variable-name registry is
[`configs/providers.yaml`](../configs/providers.yaml). Credential loading is
implemented by [`src/hsle/config.py`](../src/hsle/config.py), and provider
adapters are implemented by
[`src/hsle/providers.py`](../src/hsle/providers.py).

`HSLE_DATASET_REPO` and `HSLE_DATASET_REVISION` are configuration values, not
credentials. The release defaults to the public dataset and immutable revision
shown above; override them only to validate another explicit snapshot.

## Download the separate dataset snapshot

The public repository contains code, documentation, and configuration only.
Before running the benchmark, download the pinned public snapshot:

```bash
hsle-download-data
```

The command has a pinned public repository and revision. Environment variables
override those defaults, and explicit command-line values take precedence:

```bash
hsle-download-data \
  --repo-id "$HSLE_DATASET_REPO" \
  --revision "$HSLE_DATASET_REVISION"
```

The default destination is the Git-ignored `data/` directory. The downloaded
snapshot supplies processed question tables, correction manifests, and image
assets. For a private or gated dataset, set `HF_TOKEN`; the downloader passes
authentication through `huggingface_hub`.

## Select a response provider

Response generation uses the required `--provider` option:

```bash
python -m hsle.benchmark \
  --provider PROVIDER \
  --model PROVIDER_MODEL_ID \
  --model-id ANALYTICAL_MODEL_ID \
  --setting zero_shot \
  --output outputs/responses/model_responses.csv
```

Supported provider values are:

| `--provider` | Adapter | Credential |
|---|---|---|
| `gemini` | Gemini generate-content API | `GEMINI_API_KEY` |
| `openai` | OpenAI Responses API | `OPENAI_API_KEY` |
| `anthropic` | Anthropic Messages API | `ANTHROPIC_API_KEY` |
| `openrouter` | OpenRouter OpenAI-compatible chat-completions API | `OPENROUTER_API_KEY` |

`--model` is the provider-facing model identifier. `--model-id` is the stable
analytical identifier written to the output. If that analytical identifier is
already listed in [`configs/models.yaml`](../configs/models.yaml), its family
and modality are resolved from the registry. A new analytical identifier must
either be added to the registry or be accompanied by both `--family` and
`--modality`.

## OpenRouter

OpenRouter uses the endpoint declared in
[`configs/providers.yaml`](../configs/providers.yaml):

```text
https://openrouter.ai/api/v1
```

Set `OPENROUTER_API_KEY` in `.env`, then pass OpenRouter's provider/model route
as `--model`:

```bash
python -m hsle.benchmark \
  --provider openrouter \
  --model provider/model-name \
  --model-id stable-analysis-name \
  --family model-family \
  --modality text_only \
  --setting zero_shot \
  --output outputs/responses/stable-analysis-name.csv
```

Use `--modality multimodal` only when the selected route accepts every image
input required by the benchmark. If the stable analytical model is already in
`configs/models.yaml`, omit `--family` and `--modality`.

`OPENROUTER_SITE_URL` and `OPENROUTER_APP_NAME` are optional attribution
headers. They are not credentials.

## Judge credentials

HLE correctness and closeness judging use Gemini 3.5 Flash and therefore
require `GEMINI_API_KEY`:

```bash
python -m hsle.judge \
  --metric both \
  --input outputs/responses/model_responses.csv \
  --judge-model gemini-3.5-flash \
  --output outputs/judgments/judged_responses.csv
```

Learning from experience also requires `GEMINI_API_KEY`, even when the
evaluated model is served by OpenAI, Anthropic, or OpenRouter. The Gemini judge
evaluates each linked example attempt to generate the binary feedback shown to
the evaluated model.

## Helix local workflow

The Helix launchers do not accept credentials as command-line arguments. Set
`HSLE_GEMINI_KEY_FILE` in the process environment to a regular, non-symlink,
user-owned file with no group or other permissions:

```bash
export HSLE_GEMINI_KEY_FILE=/secure/path/gemini.key
bash script/run_helix_kimi_k26.sh
```

The launcher submits with `--export=NONE` and passes only the non-secret key
file path to the GPU worker. Preparation downloads the pinned public snapshots
without authentication. Generation reads the Gemini key from the protected
file only for at-most-once inline Gemini 3.5 Flash feedback and does not write
or print the key. Final HLE and closeness judging remains a separate
project-owner operation. See [the Helix contract](HELIX_LOCAL.md).

## Local data and generated artifacts

Downloaded benchmark data is stored beneath Git-ignored `data/`. Generated
model responses, judge outputs, aggregate metric tables, and PDF figures are
written beneath Git-ignored `outputs/`. Neither directory is included in the
public repository.

## Credential safety

- Never commit `.env`, token files, exported shell histories, provider
  response logs containing credentials, or copied key material.
- Never paste a credential into Python, YAML, notebooks, command-line
  arguments, issue text, or documentation.
- Keep only standard variable names in tracked files; secret values belong in
  `.env` or a secret manager.
- Rotate a credential immediately if it has ever appeared in a tracked file or
  repository history. Removing it from the latest commit is not sufficient.
- Before publishing, inspect both tracked files and history. `.gitignore`
  prevents new accidental additions but does not remove material already
  committed.

The release `.gitignore` excludes `.env`, `.env.*` other than
`.env.example`, and the `secrets/` and `keys/` directories.
