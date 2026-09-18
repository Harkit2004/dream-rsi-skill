"""The evaluator / task adapter contract (issue #2)."""

from pathlib import Path

import pytest

from dream_rsi.adapters.evaluator import (
    CRASH,
    OK,
    EvalResult,
    ScoreDirection,
    TaskEvaluator,
    safe_evaluate,
)
from dream_rsi.adapters.toy_evaluator import ToyEvaluator
from dream_rsi.tree import DiscoveryTree


class FakeEvaluator:
    """A minimal structural implementation of the protocol, written by hand."""

    direction = ScoreDirection.HIGHER_IS_BETTER
    baseline_score = 0.0

    def evaluate(self, artifact: str, workspace: Path) -> EvalResult:
        return EvalResult(score=float(len(artifact)), correct=True, diagnostics="fake")


class CrashingEvaluator:
    direction = ScoreDirection.HIGHER_IS_BETTER
    baseline_score = 0.0

    def evaluate(self, artifact: str, workspace: Path) -> EvalResult:
        raise RuntimeError("evaluator blew up on this artifact")


def test_fake_evaluator_satisfies_the_protocol(tmp_path):
    # A static type checker accepts the annotated binding; the isinstance check
    # is the runtime half of the same statement.
    evaluator: TaskEvaluator = FakeEvaluator()
    assert isinstance(evaluator, TaskEvaluator)
    assert evaluator.evaluate("abc", tmp_path).score == 3.0


def test_the_protocol_rejects_an_incomplete_evaluator():
    class NoDirection:
        baseline_score = 0.0

        def evaluate(self, artifact: str, workspace: Path) -> EvalResult:
            return EvalResult(score=0.0, correct=True)

    assert not isinstance(NoDirection(), TaskEvaluator)


def test_toy_evaluator_satisfies_the_protocol(tmp_path):
    evaluator: TaskEvaluator = ToyEvaluator()
    assert isinstance(evaluator, TaskEvaluator)
    result = evaluator.evaluate("def solve():\n\n    return 1\n", tmp_path)
    assert result.score == 2.0
    assert result.correct is True
    assert result.evaluated is True


def test_toy_evaluator_reports_incorrect_artifacts_without_failing(tmp_path):
    result = ToyEvaluator().evaluate("def nope():\n    return 1\n", tmp_path)
    assert result.correct is False
    # Dream-RSI §B.1: an evaluation with ``error is None`` and ``fail_class ==
    # "ok"`` is a successful evaluation even when the candidate is not valid.
    assert result.evaluated is True
    assert result.fail_class == OK
    assert result.score == 2.0


@pytest.mark.parametrize(
    ("direction", "better", "worse"),
    [
        (ScoreDirection.HIGHER_IS_BETTER, 2.0, 1.0),
        (ScoreDirection.LOWER_IS_BETTER, 1.0, 2.0),
    ],
)
def test_is_better_respects_the_declared_direction(direction, better, worse):
    assert direction.is_better(better, worse)
    assert not direction.is_better(worse, better)
    assert not direction.is_better(better, better)


@pytest.mark.parametrize("direction", list(ScoreDirection))
def test_missing_scores_are_worse_than_any_score(direction):
    assert direction.is_better(0.0, None)
    assert not direction.is_better(None, 0.0)
    assert not direction.is_better(None, None)


def test_canonical_score_is_larger_is_better():
    # Dream-RSI §3: "Scores follow a fixed task-scoring protocol, with larger
    # values indicating better quality." A lower-is-better task metric has to be
    # mapped before it is stored as ``s_v``.
    assert ScoreDirection.HIGHER_IS_BETTER.to_canonical(2.5) == 2.5
    assert ScoreDirection.LOWER_IS_BETTER.to_canonical(2.5) == -2.5
    assert ScoreDirection.LOWER_IS_BETTER.to_canonical(None) is None


def test_a_crashing_evaluator_does_not_kill_the_rollout(tmp_path):
    result = safe_evaluate(CrashingEvaluator(), "artifact", tmp_path)
    assert isinstance(result, EvalResult)
    assert result.evaluated is False
    assert result.correct is False
    assert result.score is None
    assert result.fail_class == CRASH
    assert "evaluator blew up" in result.error


def test_safe_evaluate_passes_successful_results_through(tmp_path):
    result = safe_evaluate(FakeEvaluator(), "abc", tmp_path)
    assert result.evaluated is True
    assert result.score == 3.0
    assert result.error is None


def test_eval_result_is_what_the_tree_stores(tmp_path):
    evaluator = ToyEvaluator()
    artifact = "def solve():\n    return 1\n"
    result = safe_evaluate(evaluator, artifact, tmp_path)

    tree = DiscoveryTree.with_root()
    node = tree.add_child(
        tree.root_id, artifact=artifact, **result.to_node_fields(evaluator.direction)
    )

    assert node.score == -2.0  # lower-is-better metric, stored larger-is-better
    assert node.diagnostics["raw_score"] == 2.0
    assert node.diagnostics["correct"] is True
    assert node.diagnostics["direction"] == ScoreDirection.LOWER_IS_BETTER.value
    assert node.diagnostics["fail_class"] == OK

    path = tmp_path / "tree.json"
    tree.save(path)
    assert DiscoveryTree.load(path) == tree


def test_a_failed_evaluation_stores_a_null_score(tmp_path):
    result = safe_evaluate(CrashingEvaluator(), "artifact", tmp_path)
    fields = result.to_node_fields(ScoreDirection.HIGHER_IS_BETTER)

    tree = DiscoveryTree.with_root()
    node = tree.add_child(tree.root_id, **fields)

    assert node.score is None
    assert node.diagnostics["fail_class"] == CRASH
    tree.save(tmp_path / "tree.json")  # stays JSON-serialisable
