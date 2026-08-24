"""Exact prompt templates used in generation and evaluation."""

from __future__ import annotations

from dataclasses import dataclass


ANSWER_INSTRUCTION = (
    "Give the final answer clearly. If the question has answer choices, include the option letter."
)


@dataclass(frozen=True)
class ContextExample:
    question: str
    answer: str


def zero_shot_prompt(question: str) -> str:
    return f"{ANSWER_INSTRUCTION}\n\nQUESTION:\n{question.strip()}"


def one_shot_prompt(example: ContextExample, question: str) -> str:
    return (
        f"{ANSWER_INSTRUCTION}\n\n"
        "Here is one solved example.\n\n"
        f"EXAMPLE QUESTION:\n{example.question.strip()}\n\n"
        f"EXAMPLE ANSWER:\n{example.answer.strip()}\n\n"
        f"QUESTION:\n{question.strip()}"
    )


def two_shot_prompt(examples: list[ContextExample], question: str) -> str:
    if len(examples) != 2:
        raise ValueError("two_shot_prompt requires exactly two examples")
    return (
        f"{ANSWER_INSTRUCTION}\n\n"
        "Here are two solved examples.\n\n"
        f"EXAMPLE 1 QUESTION:\n{examples[0].question.strip()}\n\n"
        f"EXAMPLE 1 ANSWER:\n{examples[0].answer.strip()}\n\n"
        f"EXAMPLE 2 QUESTION:\n{examples[1].question.strip()}\n\n"
        f"EXAMPLE 2 ANSWER:\n{examples[1].answer.strip()}\n\n"
        f"QUESTION:\n{question.strip()}"
    )


def lfe_feedback(correct: bool) -> str:
    return "Your previous answer was correct." if correct else "Your previous answer was incorrect."


# Exact prompt from the official HLE evaluation implementation.
HLE_JUDGE_PROMPT = r"""Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""


CLOSENESS_PROMPT = """You are an expert evaluator for a difficult humanities, science, mathematics, history, culture, reasoning, and multimodal question-answering benchmark.

Your task is to compare a MODEL ANSWER against a GROUND TRUTH ANSWER for the given QUESTION. You are also given the official/reference rationale when available, and the model's raw response when available.

Score how close the MODEL ANSWER is to the GROUND TRUTH ANSWER on a 0 to 10 integer scale.

Use the QUESTION to understand what type of answer is required. The questions may include short factual answers, names, dates, entities, multiple-choice answers, numeric answers, mathematical answers, scientific explanations, historical or cultural interpretation, image-based questions, and multi-step reasoning.

The main score must be based on whether the model's final answer would be accepted as correct for the question. The rationale and raw output are supporting evidence. Reward the model slightly when the reasoning is clearly in the right direction, but do not give a high score if the final answer is wrong, vague, contradictory, or missing required specificity.

QUESTION:
\"\"\"
{question}
\"\"\"

GROUND TRUTH ANSWER:
\"\"\"
{ground_truth_answer}
\"\"\"

REFERENCE RATIONALE:
\"\"\"
{ground_truth_rationale}
\"\"\"

MODEL PARSED ANSWER:
\"\"\"
{model_answer}
\"\"\"

MODEL RAW OUTPUT:
\"\"\"
{model_raw_output}
\"\"\"

MODEL EXPLANATION OR RATIONALE, IF EXTRACTED:
\"\"\"
{model_explanation}
\"\"\"

Scoring rubric:

10 = Perfect match.
The model answer is fully correct and complete. It has the same meaning as the ground truth, with no missing information, no incorrect information, and no ambiguity. Minor wording differences are acceptable.

9 = Near-perfect match.
The answer is correct and complete in substance, but has very minor wording, formatting, precision, or presentation differences that do not affect correctness. It would clearly be accepted as correct.

8 = Correct with a small imperfection.
The answer is mostly correct and would likely be accepted, but it has a small omission, slight ambiguity, minor imprecision, or extra irrelevant detail that does not materially change the answer.

7 = Largely correct but noticeably incomplete or slightly flawed.
The answer captures the main correct idea, but misses a meaningful detail, is somewhat under-specified, or contains a minor factual issue. It is closer to correct than partial.

6 = Mostly correct, but missing important detail.
The answer gets the main idea right and is more correct than incorrect. However, it is incomplete, under-specified, or missing one or more important details needed for a fully correct answer. It would receive partial credit, but not full credit.

5 = Mixed correctness.
The answer contains both correct and incorrect elements, or it answers only part of the question. It shows some understanding of the ground truth, but the missing or wrong parts are significant. It is neither clearly correct nor clearly wrong.

4 = Weakly partially correct.
The answer has a small amount of relevant correct content, but the main answer is missing, unclear, or substantially flawed. It shows some understanding but would generally not be accepted.

3 = Mostly incorrect, with a small correct element.
The answer is primarily wrong, but contains a minor relevant fragment, related concept, or partially correct clue.

2 = Very little correctness.
The answer is largely incorrect or irrelevant, with only a minimal connection to the ground truth or question.

1 = Almost entirely wrong.
The answer is incorrect, but not completely unrelated. It may mention the right topic, entity, format, or a tiny fragment related to the truth.

0 = Completely wrong.
The answer is factually wrong, contradictory to the ground truth, irrelevant to the question, empty, nonsensical, or refuses to answer when the ground truth provides an answer.

Additional rules:
- Judge meaning, not exact wording.
- Use the question to resolve ambiguity.
- Do not reward long explanations unless they support the correct answer.
- Penalize hallucinated or contradictory information, even if part of the answer is correct.
- Penalize missing required specificity. For example, if the correct answer is "Table 2" and the model says only "Table", the answer is incomplete and ambiguous.
- For multiple-choice questions, the option letter and the option meaning are equivalent if they clearly refer to the same choice.
- For numeric answers, exact equality usually deserves 10. Small rounding or formatting differences may receive 8 or 9 if acceptable in context. A different number should receive a low score unless it reflects a partially correct derivation.
- For list answers, penalize missing items, extra incorrect items, and extra wrong items. Penalize wrong ordering only when ordering matters.
- For yes/no questions, the polarity must match. The wrong polarity should receive 0 to 2 depending on whether the explanation contains any correct context.
- For image-based questions, use the text available in the question and answer. If the model answer clearly contradicts the required visual conclusion, penalize it strongly.
- Do not be overly generous. A vague answer should not receive a high score merely because it is not explicitly false.
- The final output must be exactly one integer from 0 to 10.
- Output nothing else.
"""


def render_hle_judge_prompt(question: str, response: str, correct_answer: str) -> str:
    return HLE_JUDGE_PROMPT.format(
        question=question.strip(),
        response=response.strip(),
        correct_answer=correct_answer.strip(),
    )


def render_closeness_prompt(
    question: str,
    ground_truth_answer: str,
    ground_truth_rationale: str = "",
    model_answer: str = "",
    model_raw_output: str = "",
    model_explanation: str = "",
) -> str:
    return CLOSENESS_PROMPT.format(
        question=question.strip(),
        ground_truth_answer=ground_truth_answer.strip(),
        ground_truth_rationale=ground_truth_rationale.strip(),
        model_answer=model_answer.strip(),
        model_raw_output=model_raw_output.strip(),
        model_explanation=model_explanation.strip(),
    )
