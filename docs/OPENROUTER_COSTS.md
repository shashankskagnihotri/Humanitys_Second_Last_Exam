# OpenRouter cost audit — 25 August 2026

The current zero-retry planning estimate for the five-model remainder is
**about $1,035.22 of inference**. The full-output planning reserve is
**$1,559.12**, and the controller requires a one-cent-buffered **$1,559.13**
remaining key allowance. MiniMax M2.5 is complete and is not part of this
pending workload.

Prices, route health, quantization, context, output limits, and supported
parameters were read from OpenRouter's official endpoint catalog at 12:19 UTC
on 25 August 2026. The runner repeats those unpriced checks immediately before
submission and fails closed on drift.

## Exact remaining calls

The remaining workload has 10,111 callable coordinates. A static coordinate makes one
benchmark-generation call. Each LFE coordinate makes three generation calls
and two Gemini feedback calls.

```text
generation turns = coordinates + 2 × LFE coordinates
                 = 10,111 + 2 × 2,212
                 = 14,535

Gemini feedback calls = 2 × 2,212 = 4,424
```

The requested order and exact call counts are:

| Order | Model | Callable coordinates | Generation turns | LFE feedback calls |
|---:|---|---:|---:|---:|
| 1 | Qwen 3.8 Max | 2,045 | 3,027 | 982 |
| 2 | Kimi K3 | 1,963 | 2,943 | 980 |
| 3 | Kimi K2 Thinking | 1,437 | 1,939 | 502 |
| 4 | Kimi K2.6 | 2,421 | 3,401 | 980 |
| 5 | Kimi K2.5 | 2,245 | 3,225 | 980 |
| **Total** |  | **10,111** | **14,535** | **4,424** |

There is exactly one dispatch per planned turn and no automatic retry.

## Current selected prices and costs

| Order | Exact generation route | Input / output per 1M | Historical-length estimate | 16,384-output scenario |
|---:|---|---:|---:|---:|
| 1 | Qwen 3.8 Max, Alibaba | $2.00 / $6.00 | $245.41 | $301.84 |
| 2 | Kimi K3, DeepInfra BF16 | $2.85 / $14.25 | $471.07 | $696.07 |
| 3 | Kimi K2 Thinking, Novita BF16 | $0.60 / $2.50 | $50.60 | $80.06 |
| 4 | Kimi K2.6, DeepInfra FP4 | $0.75 / $3.50 | $162.85 | $195.93 |
| 5 | Kimi K2.5, DeepInfra FP4 | $0.45 / $2.25 | $98.84 | $119.83 |
| **Generation total** |  |  | **$1,028.77** | **$1,393.73** |

The expected 4,424 Gemini 3.5 Flash Flex feedback calls add **$6.45** using
historical feedback-token means. Final HLE and closeness judging is excluded.

| Planning interpretation | Generation | LFE feedback | Combined inference |
|---|---:|---:|---:|
| Mathematical input-only floor; zero completion tokens | about $15.72 | about $2.30 | **about $18.02** |
| Lower-output case; 4,096 tokens on every model turn | about $360.23 | $6.45 expected | **about $366.68** |
| Expected; historical route-specific completion means | $1,028.77 | $6.45 expected | **$1,035.22** |
| Generation worst case; 16,384 model output, expected feedback | $1,393.73 | $6.45 expected | **$1,400.18** |
| Planning reserve; full model and feedback output allowances | $1,393.73 | $165.39 | **$1,559.12** |

These are projections, not provider-guaranteed hard dollar bounds. Exact input
token totals cannot exist before dispatch: OpenRouter records the selected
provider's tokenizer usage after each call, and LFE target prompts contain the
new model outputs and feedback from the two preceding turns. Image
tokenization, cache behavior, early failed feedback, and provider accounting
can also change the ledger. A provider/key spend cap remains the only hard
financial ceiling.

The input-only row is arithmetic, not a realistic bill prediction: a useful
response necessarily emits completion tokens, and even an unusable first
response can be billable. “Worst case” here means every model turn consumes
the configured 16,384 completion allowance while feedback stays near its
historical mean. “Planning reserve” additionally makes every feedback call
consume its full 8,192-token allowance.

The historical-length projection uses approximately 9.66 million input tokens
and 182.39 million completion tokens. These are protocol projections, not
already observed billing totals. The 16,384 scenario keeps the same projected
inputs and assumes every generation turn consumes the entire output allowance.

## Completed MiniMax M2.5 run

MiniMax's 184 safely callable coordinates were dispatched exactly once on 25
August 2026 through Novita FP8. The run produced 144 nonblank responses and 40
terminal nonresponses, made zero retries, and cost exactly **$1.45656570** as
reported by the 184 retained OpenRouter settlements. MiniMax has been removed
from the collaborator launcher; the 13 older paid/no-replay coordinates remain
excluded and are not safe to resubmit.

## Credit-purchase fee

OpenRouter separately charges a 5.5% credit-purchase fee, subject to its
minimum. The fee is not inference. Funding the $1,035.22 expected inference
entirely with new credits adds about **$56.94**, for approximately $1,092.16
cash. Funding the $1,559.13 allowance adds about **$85.75**, for approximately
$1,644.88 cash. Existing credit does not incur a second purchase fee.

## Why earlier OpenAI and Claude runs cost much less

The engineering census estimates about 0.876 million output tokens for the
completed three-model OpenAI campaign and about 2.889 million for two Claude
models. Those are LFE-adjusted estimates, not exact provider-ledger totals. The
remaining OpenRouter workload projects about **182.39 million completion
tokens**, roughly 208 times the OpenAI output volume and 63 times the Claude
output volume.

This is a protocol/output-volume difference, not a general OpenRouter markup.
The earlier OpenAI profiles used `reasoning_effort=none`, the Claude requests
did not enable extended thinking, and the remaining OpenRouter routes preserve
native or high reasoning. Kimi K3 alone contributes approximately **$471.07**
to the current historical-length estimate.

The old 4,096-token production profile was abandoned because reasoning models
could consume the allowance internally and return no visible answer. Restoring
that cap would lower the nominal estimate but would change the protocol and
recreate a known truncation failure.

## Kimi K3 endpoint successor

The v3 input archive remains byte-identical at SHA-256
`ced06f31b7d82a58db28391f6e9bf09293a88933480f6b5354784ce98d3ede5f`.
Its authenticated request metadata records the then-current Sail Research FP4
endpoint. Sail subsequently disappeared from the official endpoint catalog,
so the Git-bound runtime contract now records an explicit successor:
DeepInfra BF16 at $2.85/$14.25, 1,048,576-token context, an exact 16,384-token
completion limit, and support for reasoning, `reasoning_effort`,
`include_reasoning`, `max_tokens`, and `seed`.

Morph FP4 changed from unhealthy to healthy during the audit window and was
listed at $2.80/$14.00 at 12:19 UTC. It was not selected because its health was
volatile while DeepInfra remained healthy and exposed the exact release output
cap. The runner never silently falls back: it requests only DeepInfra BF16 and
fails before paid work if that contract changes.

## Accounting rules

- Use the exact selected endpoint's prices, not a model-level catalog minimum.
- OpenRouter includes reasoning in `completion_tokens`; never add
  `reasoning_tokens` a second time.
- A malformed or blank HTTP 2xx response may still carry billable usage even
  though it becomes terminal incorrect.
- Separate provider-ledger cost, historical-token projection, output-cap
  scenarios, and the credit-purchase fee.
- Include two Gemini feedback calls per LFE coordinate; exclude later final HLE
  and closeness judging.

## Official sources

- [MiniMax M2.5 endpoints](https://openrouter.ai/api/v1/models/minimax/minimax-m2.5/endpoints)
- [Qwen 3.8 Max endpoints](https://openrouter.ai/api/v1/models/qwen/qwen3.8-max/endpoints)
- [Kimi K3 endpoints](https://openrouter.ai/api/v1/models/moonshotai/kimi-k3/endpoints)
- [Kimi K2 Thinking endpoints](https://openrouter.ai/api/v1/models/moonshotai/kimi-k2-thinking/endpoints)
- [Kimi K2.6 endpoints](https://openrouter.ai/api/v1/models/moonshotai/kimi-k2.6/endpoints)
- [Kimi K2.5 endpoints](https://openrouter.ai/api/v1/models/moonshotai/kimi-k2.5/endpoints)
- [OpenRouter reasoning-token accounting](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
- [OpenRouter FAQ and credit-purchase fee](https://openrouter.ai/docs/faq)
