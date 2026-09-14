"""Evaluation harness for summarisation and retrieval quality.

Summarisation has no single correct output, so this measures the three things
that actually go wrong in practice rather than trying to score prose quality:

  coverage      did the briefing mention every decision the transcript contains?
  action items  were assigned tasks captured, with the right owner?
  grounding     did the RAG answer stay inside the retrieved context, or did it
                supplement from the model's own knowledge?

Grounding is the one worth having. A retrieval system that answers fluently
from parametric memory when the transcript is silent is worse than one that
says "not discussed", and nothing else in the pipeline catches it.
"""

import asyncio
import logging
import re
from dataclasses import asdict, dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CoverageScore:
    recalled: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        total = len(self.recalled) + len(self.missed)
        return len(self.recalled) / total if total else 1.0


@dataclass
class ActionItemScore:
    matched: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    owner_correct: int = 0
    owner_total: int = 0

    @property
    def recall(self) -> float:
        total = len(self.matched) + len(self.missed)
        return len(self.matched) / total if total else 1.0

    @property
    def owner_accuracy(self) -> float:
        return self.owner_correct / self.owner_total if self.owner_total else 1.0

    @property
    def score(self) -> float:
        return (self.recall + self.owner_accuracy) / 2


@dataclass
class GroundingScore:
    answered_when_present: int = 0
    total_present: int = 0
    abstained_when_absent: int = 0
    total_absent: int = 0

    @property
    def answer_rate(self) -> float:
        return self.answered_when_present / self.total_present if self.total_present else 1.0

    @property
    def abstention_rate(self) -> float:
        """The important one: how often the system correctly refuses to answer
        a question the transcript does not cover."""
        return self.abstained_when_absent / self.total_absent if self.total_absent else 1.0

    @property
    def score(self) -> float:
        return (self.answer_rate + self.abstention_rate) / 2


@dataclass
class EvalReport:
    coverage: CoverageScore
    actions: ActionItemScore
    grounding: GroundingScore

    @property
    def overall(self) -> float:
        return float(np.mean([self.coverage.score, self.actions.score, self.grounding.score]))

    def to_dict(self) -> dict:
        return {
            "overall": round(self.overall, 4),
            "coverage": round(self.coverage.score, 4),
            "action_recall": round(self.actions.recall, 4),
            "owner_accuracy": round(self.actions.owner_accuracy, 4),
            "grounding_answer_rate": round(self.grounding.answer_rate, 4),
            "grounding_abstention_rate": round(self.grounding.abstention_rate, 4),
            "detail": {
                "coverage": asdict(self.coverage),
                "actions": asdict(self.actions),
                "grounding": asdict(self.grounding),
            },
        }

    def __str__(self) -> str:
        def bar(v: float) -> str:
            return "█" * int(v * 20)

        return "\n".join([
            f"Overall                    {self.overall:.3f}",
            "",
            f"Decision coverage          {self.coverage.score:.3f}  {bar(self.coverage.score)}",
            f"Action item recall         {self.actions.recall:.3f}  {bar(self.actions.recall)}",
            f"Owner attribution          {self.actions.owner_accuracy:.3f}  "
            f"{bar(self.actions.owner_accuracy)}",
            f"Answers when present       {self.grounding.answer_rate:.3f}  "
            f"{bar(self.grounding.answer_rate)}",
            f"Abstains when absent       {self.grounding.abstention_rate:.3f}  "
            f"{bar(self.grounding.abstention_rate)}",
            "",
            f"Missed decisions:   {len(self.coverage.missed)}",
            f"Missed action items: {len(self.actions.missed)}",
        ])


# Phrases indicating the model declined to answer. Matched case-insensitively
# against the response.
ABSTENTION_MARKERS = (
    "not discussed", "does not contain", "doesn't contain", "no mention",
    "not mentioned", "not covered", "cannot answer", "can't answer",
    "not present in", "does not record", "doesn't record", "no information",
)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _mentions(haystack: str, needle: str, min_overlap: float = 0.6) -> bool:
    """Loose containment: does the output cover this expected point?

    Exact substring matching fails immediately — the summary paraphrases. This
    checks what fraction of the expected point's content words appear, which is
    crude but stable enough to detect an outright missing decision.
    """
    stop = {"the", "a", "an", "to", "of", "and", "or", "for", "in", "on",
            "at", "is", "was", "were", "be", "will", "that", "this", "with"}
    words = {w for w in _normalise(needle).split() if w not in stop and len(w) > 2}
    if not words:
        return False
    text = _normalise(haystack)
    hits = sum(1 for w in words if w in text)
    return hits / len(words) >= min_overlap


def score_coverage(briefing: str, expected_decisions: list[str]) -> CoverageScore:
    result = CoverageScore()
    for decision in expected_decisions:
        (result.recalled if _mentions(briefing, decision) else result.missed).append(decision)
    return result


def score_action_items(output: str, expected: list[dict]) -> ActionItemScore:
    """Expected items: [{"task": "...", "owner": "..."}].

    Owner accuracy is scored only over items that were found at all — an owner
    cannot be wrong on a task the summary never mentioned, and counting it
    twice would double-penalise the same miss.
    """
    result = ActionItemScore()
    for item in expected:
        task, owner = item["task"], item.get("owner")
        if not _mentions(output, task):
            result.missed.append(task)
            continue
        result.matched.append(task)
        if owner:
            result.owner_total += 1
            if _normalise(owner) in _normalise(output):
                result.owner_correct += 1
    return result


def score_grounding(results: list[dict]) -> GroundingScore:
    """Expected: [{"answer": "...", "in_transcript": bool}]."""
    score = GroundingScore()
    for item in results:
        abstained = any(m in _normalise(item["answer"]) for m in ABSTENTION_MARKERS)
        if item["in_transcript"]:
            score.total_present += 1
            if not abstained:
                score.answered_when_present += 1
        else:
            score.total_absent += 1
            if abstained:
                score.abstained_when_absent += 1
    return score


async def evaluate(
    transcript_path: str,
    ground_truth: dict,
    process_fn=None,
    answer_fn=None,
) -> EvalReport:
    """Run the pipeline over a labelled transcript and score it.

    `process_fn` and `answer_fn` are injectable so the harness can be exercised
    against stubs without spending API calls.

    Ground truth format:
      {"decisions": [...],
       "action_items": [{"task": ..., "owner": ...}],
       "questions": [{"query": ..., "in_transcript": bool}]}
    """
    if process_fn is None or answer_fn is None:
        from .pipeline import get_llm, process as _process
        from .chat import answer as _answer

        llm = get_llm()
        process_fn = process_fn or _process

        async def answer_fn(store, query):  # noqa: E306
            return await _answer(llm, store, query)

    result = await process_fn(transcript_path)

    coverage = score_coverage(result.briefing, ground_truth.get("decisions", []))
    actions = score_action_items(result.action_items, ground_truth.get("action_items", []))

    responses = []
    for q in ground_truth.get("questions", []):
        text = await answer_fn(result.store, q["query"])
        responses.append({"answer": text, "in_transcript": q["in_transcript"]})

    return EvalReport(coverage=coverage, actions=actions, grounding=score_grounding(responses))


def evaluate_sync(transcript_path, ground_truth, process_fn=None, answer_fn=None) -> EvalReport:
    return asyncio.run(evaluate(transcript_path, ground_truth, process_fn, answer_fn))
