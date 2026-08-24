# OpenRouter cost audit — 24 August 2026

The corrected planning estimate for the six-model remainder is **about
$995.70 of inference** when historical route-specific completion lengths are
used. The previously alarming figure was not caused by a general OpenRouter
price markup. It came from native-reasoning responses that are vastly longer
than the completed OpenAI and Claude campaigns, plus Kimi K3's high output
price.

Prices and route health were read from OpenRouter's official endpoint catalog
and monitored through 18:17:20 UTC on 24 August 2026. The runner independently
checks the same exact contracts immediately before submission and fails closed
on any drift.

## Workload arithmetic

The v3 runner has 10,295 callable coordinates. A static coordinate makes one
benchmark-generation call. Each LFE coordinate makes three generation calls
and two Gemini feedback calls.

```text
generation turns = coordinates + 2 × LFE coordinates
                 = 10,295 + 2 × 2,212
                 = 14,719

Gemini feedback calls = 2 × 2,212 = 4,424
```

Calls are not multiplied by six, and coordinate count is not itself the API
call count. There are six routes, but each route owns a disjoint task vector.

## Current selected prices

| Route | Generation turns | Input / output price per 1M | Historical-length estimate | 16,384-output scenario |
|---|---:|---:|---:|---:|
| Kimi K2 Thinking, Novita BF16 | 1,939 | $0.60 / $2.50 | $50.60 | $80.06 |
| Kimi 2.5, DeepInfra FP4 | 3,225 | $0.45 / $2.25 | $98.84 | $119.83 |
| Kimi 2.6, DeepInfra FP4 | 3,401 | $0.75 / $3.50 | $162.85 | $195.93 |
| Kimi K3, Sail Research FP4 | 2,943 | $2.60 / $13.00 | $429.75 | $635.01 |
| Qwen 3.8 Max, Alibaba | 3,027 | $2.00 / $6.00 | $245.41 | $301.84 |
| MiniMax M2.5, Novita FP8 | 184 | $0.30 / $1.20 | $1.80 | $3.65 |
| **Generation total** | **14,719** |  | **$989.25** | **$1,336.32** |

The expected 4,424 Gemini 3.5 Flash Flex feedback calls add **$6.45** using
historical feedback-token means. Final HLE and closeness judging is excluded.

| Planning scenario | Generation | LFE feedback | Combined inference |
|---|---:|---:|---:|
| 4,096 generated tokens on every model turn | $345.30 | $6.45 expected | **$351.75** |
| Historical route-specific completion means | $989.25 | $6.45 expected | **$995.70** |
| 16,384 generated tokens on every model turn | $1,336.32 | $6.45 expected | **$1,342.77** |
| 16,384 model output and full 8,192 feedback output | $1,336.32 | $165.39 | **$1,501.71** |

These are projections, not hard dollar bounds. Actual prompts, image
tokenization, cache behavior, provider usage reporting, and early terminal
turns can change the ledger. A provider credit/spend cap is required for a
hard financial ceiling.

The release preflight rejects free-tier keys and requires **$1,501.72** of
remaining spending allowance when the supplied inference key has a numeric
limit. This is the full-output planning reserve, not a promise that it will all
be spent or a hard maximum. OpenRouter exposes account-wide `/credits` only to
a separate management key, so an unlimited ordinary inference key cannot prove
the shared account balance. The third user must fund that balance or enable
sufficient automatic funding before launch.

OpenRouter separately charges a 5.5% credit-purchase fee, subject to its
minimum. That fee is not inference. Funding the historical estimate entirely
with newly purchased credits would add about **$54.76**, for roughly
**$1,050.46 cash**. Funding the $1,501.71 planning scenario would add about
**$82.59**, for roughly **$1,584.31 cash**.

## Why prior OpenAI and Claude runs cost much less

The project's engineering report estimates that the completed three-model
OpenAI campaign represented about 0.876 million output tokens across roughly
10,311 calls, approximately 85 tokens per call. It estimates that two Claude
models represented about 2.889 million output tokens across roughly 6,874
calls, approximately 420 tokens per call. Those are LFE-adjusted projections:
unrepresented example turns were estimated with the report's documented
multipliers, so they are not exact provider-ledger usage totals.

The unfinished OpenRouter workload projects about **183.9 million completion
tokens**, approximately 12,491 per generation call. That is about 210 times
the estimated OpenAI output volume and 64 times the estimated Claude output
volume. This is not an apples-to-apples vendor-price comparison: the earlier
OpenAI profiles used `reasoning_effort=none`, the Claude requests did not
enable extended thinking, and the remaining OpenRouter routes preserve native
or high reasoning. That protocol/output-volume difference drives most of the
gap. Kimi K3 alone contributes roughly $429.75 to the historical-length
estimate.

The old 4,096-token production profile was abandoned because reasoning models
could spend the entire allowance internally and return no visible answer.
Restoring that cap would lower the nominal estimate to $351.75 but would change
the protocol and recreate a known truncation failure. The runner therefore
keeps the scientifically established 16,384 maximum while still dispatching
only once.

## Accounting rules used

- Use the exact selected endpoint's input and output price, not a model-level
  catalog minimum.
- OpenRouter reports reasoning inside `completion_tokens`; never add
  `reasoning_tokens` a second time.
- A malformed or blank HTTP 2xx response may still carry billable usage even
  though it becomes terminal incorrect. Pre-provider 4xx rejections generally
  have no inference charge.
- Separate actual ledger cost, historical-token projection, and cap-consumed
  planning scenarios.
- Include two Gemini feedback calls per LFE coordinate; exclude later final HLE
  and closeness judging.
- Keep the 5.5% credit-purchase fee separate from inference pricing.

Cheaper endpoint tags were not substituted merely to lower the estimate. Kimi
2.6 remains on DeepInfra FP4 and MiniMax remains on Novita FP8 to preserve
continuity with their partial campaigns. Kimi K3 is the necessary exception:
the old Morph FP4 endpoint is unhealthy, while Sail Research is healthy,
advertises FP4, preserves `reasoning_effort=max`, and is slightly cheaper.
