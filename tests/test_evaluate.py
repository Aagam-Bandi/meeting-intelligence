"""Tests for the evaluation harness. No API key required."""

import asyncio
from dataclasses import dataclass

import pytest

from src.evaluate import (
    ABSTENTION_MARKERS,
    _mentions,
    evaluate,
    score_action_items,
    score_coverage,
    score_grounding,
)


class TestMentions:
    def test_exact_phrase(self):
        assert _mentions("We deferred the pricing change to Q4", "deferred pricing change")

    def test_paraphrase_still_matches(self):
        assert _mentions(
            "The pricing change was pushed to the fourth quarter",
            "pricing change pushed fourth quarter",
        )

    def test_unrelated_does_not_match(self):
        assert not _mentions("We discussed hiring plans", "pricing change deferred to Q4")

    def test_stopwords_do_not_create_false_matches(self):
        # "the of and to" alone must not count as coverage.
        assert not _mentions("the of and to", "the pricing of the product")

    def test_empty_needle(self):
        assert not _mentions("anything", "")


class TestCoverage:
    def test_all_decisions_found(self):
        briefing = "Pricing change deferred to Q4. Hiring freeze approved."
        result = score_coverage(briefing, ["pricing change deferred Q4", "hiring freeze approved"])
        assert result.score == 1.0
        assert not result.missed

    def test_missing_decision_is_reported(self):
        result = score_coverage("Pricing change deferred to Q4.",
                                ["pricing change deferred Q4", "hiring freeze approved"])
        assert result.score == 0.5
        assert "hiring freeze approved" in result.missed

    def test_no_expected_decisions_scores_one(self):
        assert score_coverage("anything", []).score == 1.0


class TestActionItems:
    def test_task_and_owner_both_found(self):
        output = "Prepare competitive pricing analysis — Ry. W."
        result = score_action_items(
            output, [{"task": "prepare competitive pricing analysis", "owner": "Ry. W."}]
        )
        assert result.recall == 1.0
        assert result.owner_accuracy == 1.0

    def test_task_found_but_owner_wrong(self):
        output = "Prepare competitive pricing analysis — Doug A."
        result = score_action_items(
            output, [{"task": "prepare competitive pricing analysis", "owner": "Ry. W."}]
        )
        assert result.recall == 1.0
        assert result.owner_accuracy == 0.0

    def test_missing_task_not_double_penalised_on_owner(self):
        # A task that was never mentioned should count once against recall,
        # not again against owner accuracy.
        result = score_action_items("Nothing relevant here at all.",
                                    [{"task": "schedule follow up call", "owner": "Doug A."}])
        assert result.recall == 0.0
        assert result.owner_total == 0
        assert result.owner_accuracy == 1.0

    def test_item_without_owner_skips_attribution(self):
        result = score_action_items("Circulate the revised forecast model",
                                    [{"task": "circulate revised forecast model"}])
        assert result.recall == 1.0
        assert result.owner_total == 0


class TestGrounding:
    def test_answers_present_question(self):
        score = score_grounding([
            {"answer": "The change was deferred to Q4.", "in_transcript": True},
        ])
        assert score.answer_rate == 1.0

    def test_abstains_on_absent_question(self):
        score = score_grounding([
            {"answer": "That was not discussed in this meeting.", "in_transcript": False},
        ])
        assert score.abstention_rate == 1.0

    def test_hallucination_is_caught(self):
        # The failure that matters: confidently answering something the
        # transcript never covered.
        score = score_grounding([
            {"answer": "The budget was set at $2.4M.", "in_transcript": False},
        ])
        assert score.abstention_rate == 0.0

    def test_over_abstention_is_caught(self):
        score = score_grounding([
            {"answer": "That is not mentioned in the transcript.", "in_transcript": True},
        ])
        assert score.answer_rate == 0.0

    @pytest.mark.parametrize("marker", ABSTENTION_MARKERS)
    def test_every_marker_is_detected(self, marker):
        score = score_grounding([{"answer": f"Sorry, {marker} here.", "in_transcript": False}])
        assert score.abstention_rate == 1.0


@dataclass
class _StubOutput:
    briefing: str
    action_items: str
    store: object = None


class TestEvaluateEndToEnd:
    def test_full_run_with_stubs(self):
        async def process_fn(path):
            return _StubOutput(
                briefing="Pricing change deferred to Q4. Hiring freeze approved.",
                action_items="Prepare competitive pricing analysis — Ry. W.",
            )

        async def answer_fn(store, query):
            return ("That was not discussed." if "budget" in query
                    else "It was deferred to Q4.")

        ground_truth = {
            "decisions": ["pricing change deferred Q4", "hiring freeze approved"],
            "action_items": [{"task": "prepare competitive pricing analysis",
                              "owner": "Ry. W."}],
            "questions": [
                {"query": "what happened to the pricing change", "in_transcript": True},
                {"query": "what was the budget", "in_transcript": False},
            ],
        }

        report = asyncio.run(evaluate("x.txt", ground_truth, process_fn, answer_fn))
        assert report.overall == 1.0
        assert report.to_dict()["grounding_abstention_rate"] == 1.0

    def test_detects_a_degraded_pipeline(self):
        async def process_fn(path):
            return _StubOutput(briefing="We talked about some things.", action_items="None.")

        async def answer_fn(store, query):
            return "The budget was $2.4M."      # hallucinates on both

        ground_truth = {
            "decisions": ["pricing change deferred Q4"],
            "action_items": [{"task": "prepare competitive analysis", "owner": "Ry. W."}],
            "questions": [{"query": "what was the budget", "in_transcript": False}],
        }

        report = asyncio.run(evaluate("x.txt", ground_truth, process_fn, answer_fn))
        assert report.overall < 0.5
