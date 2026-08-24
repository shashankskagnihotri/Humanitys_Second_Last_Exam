# Prompt protocol

This document records the generic prompt and message structures used by the
released implementation. The executable source of truth is
[`src/hsle/prompts.py`](../src/hsle/prompts.py), with message construction in
[`src/hsle/benchmark.py`](../src/hsle/benchmark.py).

No system message is used for response generation, HLE correctness judging, or
closeness judging. Generation uses only `user` and `assistant` roles. Dynamic
question, answer, rationale, and response fields are stripped of leading and
trailing whitespace before insertion.

The public repository contains code, documentation, and configuration only.
The benchmark dataset, correction manifests, and image assets must be
downloaded separately before running the pipeline; they materialize beneath
the Git-ignored `data/` directory. Generated model responses, judge outputs,
aggregate metric tables, and PDF figures are written beneath the Git-ignored
`outputs/` directory. Neither input data nor generated artifacts are included
in the public repository.

## Common answer instruction

Every question presented to an evaluated model begins with this exact text:

```text
Give the final answer clearly. If the question has answer choices, include the option letter.
```

In the templates below, braces denote a dynamic field; they are not included
in a rendered prompt.

## Zero-shot

Zero-shot is one user message:

```text
Give the final answer clearly. If the question has answer choices, include the option letter.

QUESTION:
{TARGET_QUESTION}
```

For a multimodal target, the target image or images are attached to this same
user message.

## One-shot

One-shot is one user message:

```text
Give the final answer clearly. If the question has answer choices, include the option letter.

Here is one solved example.

EXAMPLE QUESTION:
{EXAMPLE_QUESTION}

EXAMPLE ANSWER:
{EXAMPLE_ANSWER}

QUESTION:
{TARGET_QUESTION}
```

Each target has two fixed linked examples. The complete one-shot condition is
therefore run twice:

- `one_shot_a` uses linked example 1.
- `one_shot_b` uses linked example 2.

For a multimodal model, image parts are attached in example-then-target order.
The two responses are retained separately at the response layer and averaged
within the same model-question pair before the one-shot aggregate is computed.

## Two-shot

Two-shot is one user message:

```text
Give the final answer clearly. If the question has answer choices, include the option letter.

Here are two solved examples.

EXAMPLE 1 QUESTION:
{EXAMPLE_1_QUESTION}

EXAMPLE 1 ANSWER:
{EXAMPLE_1_ANSWER}

EXAMPLE 2 QUESTION:
{EXAMPLE_2_QUESTION}

EXAMPLE 2 ANSWER:
{EXAMPLE_2_ANSWER}

QUESTION:
{TARGET_QUESTION}
```

For a multimodal model, image parts are attached in example-1,
example-2, then target order.

## Learning from experience

Learning from experience is an interactive conversation. Both linked examples
are attempted by the evaluated model; each attempt is judged against that
example's answer with the HLE correctness evaluator. Only binary feedback is
shown to the evaluated model. Neither the reference answer nor the judge's
reasoning is inserted into the conversation.

The conceptual sequence is:

```text
USER:
{ZERO_SHOT_PROMPT_FOR_EXAMPLE_1}

ASSISTANT:
{MODEL_RESPONSE_TO_EXAMPLE_1}

USER:
{BINARY_FEEDBACK_1}

USER:
{ZERO_SHOT_PROMPT_FOR_EXAMPLE_2}

ASSISTANT:
{MODEL_RESPONSE_TO_EXAMPLE_2}

USER:
{BINARY_FEEDBACK_2}

USER:
{ZERO_SHOT_PROMPT_FOR_TARGET}
```

The binary feedback string is exactly one of:

```text
Your previous answer was correct.
```

```text
Your previous answer was incorrect.
```

The implementation coalesces adjacent messages with the same role. The actual
API message topology used for the final target generation is therefore:

```text
1. user:      zero-shot prompt for example 1
2. assistant: model response to example 1
3. user:      feedback 1 + "\n\n" + zero-shot prompt for example 2
4. assistant: model response to example 2
5. user:      feedback 2 + "\n\n" + zero-shot prompt for the target
```

For multimodal models, each example's image parts accompany its example
question, and the target image parts accompany the final target question.

## HLE correctness judge

`HLE_JUDGE_PROMPT` in
[`src/hsle/prompts.py`](../src/hsle/prompts.py) is the complete, exact
evaluation template. It is the prompt from the pinned official HLE evaluation
implementation:

- source file: `hle_eval/run_judge_results.py`
- pinned revision: `26dca2e253b405105b4c3d8c2f5af06f86f90c66`
- exact template size: 1,290 UTF-8 bytes
- exact template SHA-256:
  `0f0023ee579b8c134f1834ed8952778b9e01460e31d47c242ee3629da9d44835`

The renderer inserts `question`, `response`, and `correct_answer`. The released
judge requests structured JSON with the extracted final answer, reasoning,
`yes`/`no` correctness, integer confidence from 0 through 100, and a strict
validation flag. The response must pass the schema in
[`src/hsle/judge.py`](../src/hsle/judge.py).

## Closeness judge

`CLOSENESS_PROMPT` in
[`src/hsle/prompts.py`](../src/hsle/prompts.py) is the complete, exact
project-defined rubric and template:

- exact template size: 5,323 UTF-8 bytes
- exact template SHA-256:
  `61eabb9aadb49f1cd99c0ab5b5815ffd9b095aaecd1e41110be16eacef328226`

It inserts these six fields:

1. question;
2. corrected ground-truth answer;
3. corrected reference rationale;
4. parsed model answer;
5. raw model output; and
6. extracted model explanation or rationale.

The full source-controlled rubric defines every integer from 0 through 10 and
requires the output to contain exactly one integer. The parser rejects any
other output.

## Judge model and decoding configuration

Both outcome layers use `gemini-3.5-flash`.

| Outcome | Candidates | Seed | Reasoning level | Output limit | Accepted output |
|---|---:|---:|---|---:|---|
| HLE correctness | 1 | 0 | Minimal | 65,536 tokens | Validated JSON schema |
| Closeness | 1 | 0 | Medium | 8,192 tokens | One integer in `[0, 10]` |

The exact configurations are implemented in
[`src/hsle/judge.py`](../src/hsle/judge.py).

## Verifying the prompt hashes

Run this from the repository root after installing the package:

```bash
python - <<'PY'
import hashlib
from hsle.prompts import CLOSENESS_PROMPT, HLE_JUDGE_PROMPT

for name, prompt in (
    ("HLE_JUDGE_PROMPT", HLE_JUDGE_PROMPT),
    ("CLOSENESS_PROMPT", CLOSENESS_PROMPT),
):
    print(name, hashlib.sha256(prompt.encode("utf-8")).hexdigest())
PY
```

Any change in whitespace, escaping, punctuation, or a trailing newline changes
the corresponding digest.
