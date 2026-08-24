# Ground-truth corrections

The release keeps corrections as a versioned layer with explicit old values,
new values, decisions, evidence, and content hashes. Reproduction must use the
pinned revisions and manifests below rather than an unversioned current
snapshot.

The public repository contains no dataset rows, image assets, correction
tables, model responses, judgments, aggregate metrics, or figures. The
benchmark data and correction provenance are distributed as a separate
Hugging Face dataset snapshot. Generated artifacts are written beneath the
Git-ignored `outputs/` directory and are not included.

## Download the data first

The downloader defaults to the public dataset and immutable release revision.
Run:

```bash
hsle-download-data
```

The default is `shashankskagnihotri/humanitys-second-last-exam` at revision
`582207e5bd95b4f4e2948887c2d613398e98a17e`. `HSLE_DATASET_REPO` and
`HSLE_DATASET_REVISION` can select another explicit snapshot. The command
materializes the snapshot beneath the Git-ignored `data/` directory by
default. A private or gated alternative also requires `HF_TOKEN`; see
[`CREDENTIALS.md`](CREDENTIALS.md).

All data paths below are snapshot-relative paths. With the default destination,
for example, snapshot path `corrections/example.csv` materializes locally as
`data/corrections/example.csv`.

## Exact correction counts

The target set contains 491 questions. Component-level corrections are:

| Corrected component | Target questions |
|---|---:|
| Question | 22 |
| Answer | 93 |
| Answer, semantic change | 91 |
| Answer, normalization-only change | 2 |
| Rationale | 160 |
| Any target component | 172 |

Nine targets have both a question correction and an answer correction. The
component counts overlap and therefore must not be added to infer the number
of unique corrected targets.

One additional context-example row has corrected question, answer, and
rationale components. It is separate from the 491 target-question counts.

## Stable identifiers and exact old/new tables

`original_question_id` is the stable target identifier. Exact target IDs and
values are available in these compact tables inside the downloaded snapshot:

- `corrections/question_correction_audit.csv` contains exactly 22 records. It
  includes the target ID, exact old and new question text, selected-value
  source, evidence-source identifiers, source relation, and correction
  decision.
  SHA-256:
  `db0bb3c9e629e1b2de66c72f84963bb45c1b6cd003858bfe8b92bef1e5b6b359`.
- `corrections/answer_correction_audit.csv` contains exactly 93 records. It
  includes the target ID, corrected question, exact old and new answers, the
  semantic-change flag, selected-value source, evidence-source identifiers,
  source relation, conflict flag, and correction decision.
  SHA-256:
  `143086b40245de21151fce9611a297fc54b4c085ad4eaf49d88a39429a6745af`.

The full row-level audit is
`corrections/hle_491_correction_manifest.csv` inside the snapshot. It contains
one record for each of the 491 targets and 74 provenance fields. For every
component it retains the local value, audited source value, selected corrected
value, application flag, evidence identifiers, relation to the secondary
source, and decision. Its SHA-256 is
`1699f779058d58963d7664b54d3217e7fb8f62197da3c48e633cdb5bbaa57b38`.

The rationale changes are represented exactly in the full manifest. Select
rows where `apply_rationale_correction` is `True` and read
`rationale_old_value`, `rationale_new_value`,
`rationale_applied_value_source_identifier`,
`rationale_decision_evidence_source_identifiers`, and
`correction_decision`. This selection contains exactly 160 rows.

The 91 semantic and two normalization-only answer changes are distinguished by
the combination of `apply_answer_correction` and `semantic_answer_change`:

```text
semantic:           apply_answer_correction=True and semantic_answer_change=True
normalization-only: apply_answer_correction=True and semantic_answer_change=False
```

## Pinned evidence sources

The correction audit pins these dataset identifiers and revisions:

| Dataset identifier | Revision |
|---|---|
| `skylenage-ai/HLE-Verified` | `0bc83643672d4f68a5f89998617a639d85e7318b` |
| `futurehouse/hle-gold-bio-chem` | `1feb9e1d545731dba81e594438330406830e5260` |

The processed-data metadata also records the upstream `cais/hle` source at
revision `5a81a4c7271a2a2a312b9a690f0c2fde837e4c29`.

The source identifiers, component decisions, conflicts, and inference
boundaries are recorded per row. A secondary-source disagreement is not by
itself treated as an automatic correction; the selected values are those
explicitly marked by the manifest's application fields.

Machine-readable audit metadata is stored at snapshot path
`corrections/hle_491_correction_metadata.json`. It records the counts above,
source intersections and conflicts, component content hashes, and hashes of
the exact corrected target-ID sets.

## Corrected context example

The single corrected context row, including its stable identifiers, exact old
and new question/answer/rationale payloads, component evidence, and row hashes,
is fully specified at snapshot path
`corrections/hsle_context_correction_manifest.json`. The correction is marked
`manual-scientific-audit`; the two pinned external datasets are recorded as
supporting invalidation evidence because they publish target records rather
than context-example corrections.

The context manifest SHA-256 is
`6a4eeea708fa354441bb08e2e4a917d71d489e6464b9e6bc2c332ea03ecb52e8`.

## Applied processed data

The corrected target and context records used by the benchmark are stored at
these snapshot-relative paths:

| File | Records | SHA-256 |
|---|---:|---|
| `processed/hsle_original_questions.csv` | 491 targets | `4e6eb4ec7610493c7b1f57c9406e40d3874f6fdd55bb10d6bf5e92b72524b052` |
| `processed/hsle_context_examples.csv` | 982 context examples | `8b886f7d2e593932a030526f24af4635f76760bb1e96f73bca8613500a6a3bb4` |
| `processed/hsle_question_example_links.csv` | 491 target-to-example links | `83c71e609f8bb1ae2c6b35c2680bc4d1059cb3def9535472f3f3e343e0ba249f` |
| `processed/hsle_image_manifest.csv` | 491 target image records | `d3632f5aed3ea61f38888cd9747ce85addea067965264691868e4e7d8afb8fec` |
| `processed/hsle_all_rows.csv` | 1,473 combined rows | `891ead1a859f5bd70f670a69cfcbd57be4fd14746fb49f6b208eef6a4c3c9b38` |

Snapshot path `processed/dataset_metadata.json` binds these processed artifacts
to the correction manifests and pinned source revisions. Image assets used by
multimodal questions are under snapshot path `images/`.
